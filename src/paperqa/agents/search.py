"""文档发现与版本化索引。查询始终使用开始时打开的快照。"""

import asyncio
import gzip
import json
import pickle
import zlib
from pathlib import Path
from uuid import uuid4

import numpy as np
import tantivy

from paperqa.catalog import Catalog
from paperqa.llms import NumpyVectorStore
from paperqa.readers import read_doc
from paperqa.retrieval import (
    ChunkKeywordIndex,
    build_vectors,
    embedding_identity,
    fuse_rankings,
    lexical_query,
    scoped_query,
)
from paperqa.types import Doc, Text
from paperqa.utils import atomic_write, dump_json, stable_id


def _read_asset(directory, entry):
    data = json.loads(gzip.decompress((directory / entry["parsed"]).read_bytes()))
    data["doc"].update(
        docname=entry["name"], citation=entry["name"],
        source_path=str(directory / entry["original"]),
    )
    doc = Doc.model_validate(data["doc"])
    return doc, [Text.model_validate({**text, "doc": doc}) for text in data["texts"]]


def _write_asset(path, payload):
    atomic_write(path, gzip.compress(dump_json(payload).encode("utf-8")))


class SearchIndex:
    def __init__(self, settings, catalog, generation, manifest):
        self.settings = settings
        self.catalog = catalog
        self.directory = catalog.directory
        self.generation = generation
        self.manifest = manifest
        self.path = self.directory / "versions" / generation if generation else self.directory
        self._documents = {}
        for path, entry in sorted(manifest["files"].items()):
            self._documents.setdefault(entry["doc_id"], {"name": Path(path).name, **entry})
        self._parsed = {}
        self._index = tantivy.Index.open(str(self.path / "index"))
        self.keywords = None
        self.vectors = None
        self.build_stats = {"parsed": 0, "encoded": 0, "reused_version": True}

    @property
    def doc_ids(self):
        return set(self._documents)

    def document_list(self):
        return [
            {"doc_id": doc_id, "name": entry["name"]}
            for doc_id, entry in self._documents.items()
        ]

    def get_saved_object(self, doc_id):
        if doc_id not in self._documents:
            raise ValueError("未知文档 ID")
        if doc_id not in self._parsed:
            if self.generation:
                item = _read_asset(self.directory, self._documents[doc_id])
            else:
                # 旧版只打开已登记的本地缓存。新版解析产物使用 JSON。
                path = self.directory / "docs" / (doc_id + ".zip")
                data = pickle.loads(zlib.decompress(path.read_bytes()))
                doc = Doc.model_validate(data["doc"])
                item = doc, [Text.model_validate(t) for t in data["texts"]]
            self._parsed[doc_id] = item
        return self._parsed[doc_id]

    def all_texts(self):
        return [
            text for doc_id in sorted(self.doc_ids) for text in self.get_saved_object(doc_id)[1]
        ]

    async def open_retrieval(self):
        if not self.manifest.get("chunks"):
            return
        texts = await asyncio.to_thread(self.all_texts)
        self.keywords = await asyncio.to_thread(
            ChunkKeywordIndex, texts, self.path / "chunks", build=False
        )
        if self.manifest.get("embedding") != embedding_identity(self.settings):
            return
        matrix = await asyncio.to_thread(np.load, self.path / "vectors.npy", allow_pickle=False)
        if matrix.ndim != 2 or len(matrix) != len(texts) or not np.isfinite(matrix).all():
            raise ValueError("索引版本的片段与向量不匹配")
        self.vectors = NumpyVectorStore(self.settings.answer.mmr_lambda)
        await self.vectors.add_texts_and_embeddings(
            text.model_copy(update={"embedding": vector.tolist()})
            for text, vector in zip(texts, matrix, strict=True)
        )

    def query(self, query, count=3, allowed_doc_ids=None):
        allowed = self.doc_ids if allowed_doc_ids is None else self.doc_ids & set(allowed_doc_ids)
        if not allowed:
            return []
        searcher = self._index.searcher()
        if self.generation:
            parsed = scoped_query(self._index, lexical_query(self._index, query), allowed)
        else:
            parsed = self._index.parse_query(query, ["body"])
        hits = searcher.search(parsed, limit=len(self._documents)).hits
        result = []
        for score, address in hits:
            doc_id = searcher.doc(address).to_dict()["doc_id"][0]
            if doc_id in allowed:
                result.append({"doc_id": doc_id, "score": score})
        return sorted(result, key=lambda x: (-x["score"], x["doc_id"]))[:count]

    async def discover(self, question, count, allowed):
        use_dense = self.settings.index.document_hybrid and self.vectors is not None
        lexical = await asyncio.to_thread(
            self.query, question, count * 3 if use_dense else count, allowed
        )
        if not use_dense:
            return lexical
        try:
            chunks, _ = await self.vectors.similarity_search(
                question, max(48, count * 12), self.settings.get_embedding_model(), allowed
            )
        except (OSError, RuntimeError, ValueError):
            return lexical[:count]
        dense = list(dict.fromkeys(t.doc.dockey for t in chunks))
        ranked = fuse_rankings(
            [hit["doc_id"] for hit in lexical], dense,
            order={doc_id: i for i, doc_id in enumerate(sorted(self.doc_ids))},
        )
        return [{"doc_id": doc_id, "score": score} for doc_id, score in ranked[:count]]


def _build_keywords(path, documents, texts):
    path.mkdir(parents=True)
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("doc_id", stored=True, tokenizer_name="raw")
    builder.add_text_field("body")
    index = tantivy.Index(builder.build(), path=str(path))
    writer = index.writer()
    for doc_id in sorted(documents):
        doc, chunks = documents[doc_id]
        writer.add_document(tantivy.Document(
            doc_id=doc_id, body=doc.docname + "\n" + "\n".join(t.text for t in chunks)
        ))
    writer.commit()
    writer.wait_merging_threads()
    if texts is not None:
        chunk_path = path.parent / "chunks"
        chunk_path.mkdir()
        ChunkKeywordIndex(texts, chunk_path)


async def _collect_documents(settings, directory, root, incoming):
    files = {
        p for p in root.rglob("*")
        if p.suffix.lower() in {".pdf", ".txt"} and not p.is_relative_to(directory) and p.is_file()
    }
    # 解析设置改变时写入另一个地址，旧查询仍能读取自己的解析产物。
    parse_key = stable_id(dump_json(["parse-v2", settings.parsing.model_dump()]))
    entries, parsed = {}, 0
    for path in sorted(files | incoming.keys()):
        data = incoming[path] if path in incoming else await asyncio.to_thread(path.read_bytes)
        doc_id = stable_id(data)
        original = f"sources/{doc_id}{path.suffix.lower()}"
        cached = f"parsed/{parse_key}/{doc_id}.json.gz"
        name = path.name.removeprefix(doc_id + "-")
        entries[str(path)] = {
            "doc_id": doc_id, "name": name, "original": original, "parsed": cached
        }
        blob = directory / original
        if not blob.is_file():
            await asyncio.to_thread(atomic_write, blob, data)
        if not (directory / cached).is_file():
            doc = Doc(
                dockey=doc_id, docname=name, citation=name, content_hash=doc_id,
                source_path=str(blob),
            )
            texts = await read_doc(blob, doc, settings)
            payload = {"doc": doc.model_dump(), "texts": [
                t.model_dump(exclude={"doc", "embedding"}) for t in texts
            ]}
            await asyncio.to_thread(_write_asset, directory / cached, payload)
            parsed += 1
    return entries, parsed


def _load_documents(directory, entries):
    docs = {}
    for entry in entries.values():
        if entry["doc_id"] not in docs:
            docs[entry["doc_id"]] = _read_asset(directory, entry)
    return docs


def _publish(index, previous, job, incoming):
    created = []
    try:
        for path, data in incoming.items():
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError("上传路径已存在不同内容")
            else:
                atomic_write(path, data)
                created.append(path)
        index.catalog.publish(index.generation, index.manifest, previous, job)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


async def get_directory_index(
    settings, build=False, *, catalog=None, current=None, job=None, incoming=None
):
    catalog = catalog or await asyncio.to_thread(Catalog, settings.index)
    generation, manifest = await asyncio.to_thread(catalog.active)
    root = Path(settings.index.paper_directory).resolve()
    parsing = settings.parsing.model_dump()
    if not build:
        if manifest is None:
            legacy = catalog.directory / "files.json"
            if not legacy.is_file():
                raise FileNotFoundError("尚无索引，请先运行 pqa index")
            manifest = json.loads(legacy.read_text(encoding="utf-8"))
            if manifest.get("format") != 1:
                raise ValueError("旧索引格式不匹配，请重建")
        if manifest.get("paper_directory") != str(root) or manifest.get("parsing") != parsing:
            raise ValueError("资料目录或解析配置已变更，请重建索引")
        if generation and current and current.generation == generation:
            return current
        index = await asyncio.to_thread(SearchIndex, settings, catalog, generation, manifest)
        await index.open_retrieval()
        return index

    if not root.is_dir():
        raise ValueError(f"材料目录不存在：{root}")
    incoming = incoming or {}
    files, parsed = await _collect_documents(settings, catalog.directory, root, incoming)
    hybrid = settings.answer.retrieval_mode == "hybrid"
    local = settings.embedding == "sparse" or settings.embedding.startswith("local/")
    signature = embedding_identity(settings) if hybrid and local else None
    target = {
        "format": 2, "files": files, "parsing": parsing, "paper_directory": str(root),
        "chunks": hybrid, "embedding": signature,
    }
    if generation and target == manifest:
        index = await get_directory_index(settings, catalog=catalog, current=current)
        if index.generation != generation:
            raise ValueError("构建期间索引版本已更新，请重试入库")
        await asyncio.to_thread(_publish, index, generation, job, incoming)
        index.build_stats = {"parsed": parsed, "encoded": 0, "reused_version": True}
        return index

    next_id = uuid4().hex
    path = catalog.directory / "versions" / next_id
    documents = await asyncio.to_thread(_load_documents, catalog.directory, files)
    texts = [text for doc_id in sorted(documents) for text in documents[doc_id][1]]
    await asyncio.to_thread(_build_keywords, path / "index", documents, texts if hybrid else None)
    encoded = 0
    if signature:
        matrix, encoded = await build_vectors(
            texts, settings.get_embedding_model(), catalog.directory / "vectors" / signature
        )
        await asyncio.to_thread(np.save, path / "vectors.npy", matrix, allow_pickle=False)
    # 发布前验证查询方能用这些同版本的产物打开索引。
    index = await asyncio.to_thread(SearchIndex, settings, catalog, next_id, target)
    await index.open_retrieval()
    index.build_stats = {"parsed": parsed, "encoded": encoded, "reused_version": False}
    await asyncio.to_thread(_publish, index, generation, job, incoming)
    return index
