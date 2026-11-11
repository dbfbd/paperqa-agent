"""Local semantic embeddings and scoped chunk-level lexical fusion."""

import asyncio
import json
import re
from functools import lru_cache
from io import BytesIO

import numpy as np
import tantivy

from paperqa.utils import atomic_write, stable_id


def embedding_identity(settings):
    return stable_id(json.dumps(
        ["vectors-v1", settings.embedding, settings.embedding_params], sort_keys=True, default=str
    ))


@lru_cache(maxsize=2)
def _load_embedding(source, revision, device, max_length, offline, threads):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads)
    model = SentenceTransformer(
        source, revision=revision, device=device, local_files_only=offline
    )
    model.max_seq_length = min(model.max_seq_length, max_length)
    return model


class LocalEmbedding:
    """Optional local backend; query/document prefixes are explicit, not guessed."""

    def __init__(self, name, config):
        self.name = name
        self.config = config

    def _encode(self, texts, prefix):
        p = self.config
        model = _load_embedding(
            p.get("path") or self.name,
            p.get("revision"),
            p.get("device", "cpu"),
            p.get("max_length", 512),
            p.get("local_files_only", False),
            p.get("threads", 6),
        )
        return model.encode(
            [prefix + t for t in texts], batch_size=p.get("batch_size", 16),
            normalize_embeddings=True, show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

    async def embed_documents(self, texts):
        return await asyncio.to_thread(
            self._encode, texts, self.config.get("document_prefix", "")
        )

    async def embed_document(self, text):
        vectors = await asyncio.to_thread(
            self._encode, [text], self.config.get("query_prefix", "")
        )
        return vectors[0]


def lexical_tokens(question):
    return list(dict.fromkeys(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", question)))[:80]


def lexical_query(index, question):
    terms = lexical_tokens(question)
    if not terms:
        return tantivy.Query.empty_query()
    return index.parse_query(" OR ".join(f'"{term}"' for term in terms), ["body"])


def scoped_query(index, query, doc_ids):
    if not doc_ids:
        return tantivy.Query.empty_query()
    scope = tantivy.Query.term_set_query(index.schema, "doc_id", sorted(doc_ids))
    return tantivy.Query.boolean_query(
        [(tantivy.Occur.Must, query), (tantivy.Occur.Must, scope)]
    )


class ChunkKeywordIndex:
    def __init__(self, texts, path=None, *, build=True):
        self.texts = list(texts)
        self.order = {text.chunk_id: i for i, text in enumerate(self.texts)}
        if not build:
            self.index = tantivy.Index.open(str(path))
            return
        builder = tantivy.SchemaBuilder()
        builder.add_integer_field("position", stored=True)
        builder.add_text_field("doc_id", tokenizer_name="raw")
        builder.add_text_field("body", tokenizer_name="en_stem")
        self.index = tantivy.Index(builder.build(), path=str(path) if path is not None else None)
        writer = self.index.writer()
        for position, text in enumerate(self.texts):
            writer.add_document(tantivy.Document(
                position=position, doc_id=text.doc.dockey, body=text.text
            ))
        writer.commit()
        writer.wait_merging_threads()
        self.index.reload()

    def search(self, question, scope, limit):
        if not scope or not self.texts or not lexical_tokens(question):
            return []
        query = scoped_query(self.index, lexical_query(self.index, question), scope)
        searcher = self.index.searcher()
        # 过滤进入索引查询，避免无关文档占用返回名额。
        # 取全匹配后按位置稳定处理同分项，适合小型本地资料库。
        hits = searcher.search(query, limit=len(self.texts)).hits
        ranked = [(score, int(searcher.doc(address).to_dict()["position"][0]))
                  for score, address in hits]
        return [
            self.texts[position]
            for _, position in sorted(ranked, key=lambda pair: (-pair[0], pair[1]))[:limit]
        ]


def fuse_rankings(*rankings, order):
    scores = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0.0) + 1 / (60 + rank)
    chosen = sorted(scores, key=lambda key: (-scores[key], order[key]))
    return [(key, scores[key]) for key in chosen]


async def hybrid_candidates(store, keywords, question, count, model, scope):
    """保留 Top-k 截断前的候选，供按需补查和离线归因使用。"""
    dense, _ = await store.similarity_search(question, count * 3, model, scope)
    lexical = await asyncio.to_thread(keywords.search, question, scope, count * 3)
    lookup = {t.chunk_id: t for t in [*dense, *lexical]}
    ranked = fuse_rankings(
        [t.chunk_id for t in dense], [t.chunk_id for t in lexical],
        order=keywords.order,
    )
    return [(lookup[key], score) for key, score in ranked]


async def hybrid_search(store, keywords, question, count, model, scope):
    ranked = (await hybrid_candidates(store, keywords, question, count, model, scope))[:count]
    return [text for text, _ in ranked], [score for _, score in ranked]


def _load_vector(path):
    try:
        vector = np.load(path, allow_pickle=False)
        if vector.ndim == 1 and vector.size and np.isfinite(vector).all():
            return vector.astype(np.float32, copy=False)
    except (OSError, EOFError, ValueError):
        pass
    return None


def _write_vector(path, vector):
    buffer = BytesIO()
    np.save(buffer, vector, allow_pickle=False)
    atomic_write(path, buffer.getvalue())


async def build_vectors(texts, model, directory):
    """只编码持久缓存里缺失的文本；重复文本仍保留各自的来源。"""
    keys = [stable_id(text.text) for text in texts]
    inputs = dict(zip(keys, (text.text for text in texts), strict=True))
    vectors = await asyncio.to_thread(
        lambda: {key: _load_vector(directory / (key + ".npy")) for key in inputs}
    )
    missing = [key for key, value in vectors.items() if value is None]
    for start in range(0, len(missing), 128):
        batch = missing[start : start + 128]
        values = np.asarray(
            await model.embed_documents([inputs[key] for key in batch]), dtype=np.float32
        )
        if (values.ndim != 2 or len(values) != len(batch) or not values.shape[1]
                or not np.isfinite(values).all()):
            raise ValueError("编码返回的向量数量、维度或数值无效")
        for key, vector in zip(batch, values, strict=True):
            vectors[key] = vector

    matrix = await asyncio.to_thread(
        lambda: np.stack([vectors[key] for key in keys]) if keys else np.empty((0, 0))
    )

    def persist():
        for key in missing:
            _write_vector(directory / (key + ".npy"), vectors[key])

    await asyncio.to_thread(persist)
    return matrix, len(missing)
