"""从原 llms.py 精简的 NumPy 检索与 MMR；范围过滤先于 top-k。"""

from abc import ABC, abstractmethod

import numpy as np


def embedding_model_factory(name="sparse", config=None):
    if name.startswith("local/"):
        from paperqa.retrieval import LocalEmbedding

        return LocalEmbedding(name.removeprefix("local/"), config or {})
    from lmi import LiteLLMEmbeddingModel, SparseEmbeddingModel

    if name == "sparse":
        return SparseEmbeddingModel(name="sparse")
    return LiteLLMEmbeddingModel(name=name, config={"kwargs": config or {}})


def cosine_similarity(a, b):
    a, b = np.atleast_2d(a).astype(float), np.atleast_2d(b).astype(float)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T


class VectorStore(ABC):
    def __init__(self, mmr_lambda=0.8):
        self.mmr_lambda = mmr_lambda
        self.texts_hashes: set[str] = set()

    def __len__(self):
        return len(self.texts_hashes)

    def __contains__(self, text):
        return text.chunk_id in self.texts_hashes

    @abstractmethod
    async def add_texts_and_embeddings(self, texts): ...

    @abstractmethod
    async def similarity_search(self, query, k, embedding_model, doc_ids=None): ...

    @abstractmethod
    def clear(self): ...

    async def max_marginal_relevance_search(
        self, query, k, fetch_k, embedding_model, doc_ids=None
    ):
        if fetch_k < k:
            raise ValueError("fetch_k 不得小于 k")
        texts, scores = await self.similarity_search(query, fetch_k, embedding_model, doc_ids)
        if len(texts) <= k or self.mmr_lambda >= 1:
            return texts[:k], scores[:k]
        similarities = cosine_similarity(
            [t.embedding for t in texts], [t.embedding for t in texts]
        )
        selected = [0]
        while len(selected) < k:
            values = self.mmr_lambda * np.array(scores) - (1 - self.mmr_lambda) * similarities[
                :, selected
            ].max(axis=1)
            values[selected] = -np.inf
            selected.append(int(values.argmax()))
        return [texts[i] for i in selected], [scores[i] for i in selected]


class NumpyVectorStore(VectorStore):
    def __init__(self, mmr_lambda=0.8):
        super().__init__(mmr_lambda)
        self.texts = []

    async def add_texts_and_embeddings(self, texts):
        for text in texts:
            if text.embedding is None:
                raise ValueError("加入索引前必须生成向量")
            if text not in self:
                self.texts.append(text)
                self.texts_hashes.add(text.chunk_id)

    async def similarity_search(self, query, k, embedding_model, doc_ids=None):
        allowed = None if doc_ids is None else set(doc_ids)
        candidates = [t for t in self.texts if allowed is None or t.doc.dockey in allowed]
        if not candidates:
            return [], []
        embedding = await embedding_model.embed_document(query)
        scores = cosine_similarity([embedding], [t.embedding for t in candidates])[0]
        chosen = np.argsort(-scores, kind="stable")[:k]
        return [candidates[i] for i in chosen], [float(scores[i]) for i in chosen]

    def clear(self):
        self.texts.clear()
        self.texts_hashes.clear()
