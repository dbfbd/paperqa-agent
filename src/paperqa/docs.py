"""任务内材料工作区；取证、回查与回答共用来源标识。"""

import asyncio
import re
from pathlib import Path

from paperqa import comparison, prompts
from paperqa.core import map_fxn_summary, request_model
from paperqa.llms import NumpyVectorStore
from paperqa.readers import read_doc
from paperqa.retrieval import ChunkKeywordIndex, embedding_identity, hybrid_candidates
from paperqa.types import Doc, SourceRead
from paperqa.utils import dump_json, stable_id


class Docs:
    def __init__(self, settings=None):
        from paperqa.settings import Settings

        self.settings = settings or Settings()
        self.docs: dict[str, Doc] = {}
        self.texts = []
        self.index = NumpyVectorStore(self.settings.answer.mmr_lambda)
        self._embedding_model = None
        self._embedding_signature = self._embedding_identity()
        self._keyword_index = None
        self.snapshot = None
        self.retrieval_cache = None

    def _embedding_identity(self):
        return embedding_identity(self.settings)

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = self.settings.get_embedding_model()
        return self._embedding_model

    async def aadd(self, path):
        path = Path(path).resolve()
        content_hash = await asyncio.to_thread(lambda: stable_id(path.read_bytes()))
        if content_hash in self.docs:
            return self.docs[content_hash]
        doc = Doc(
            dockey=content_hash,
            content_hash=content_hash,
            docname=path.name,
            citation=path.name,
            source_path=str(path),
        )
        texts = await read_doc(path, doc, self.settings)
        self.aadd_texts(doc, texts)
        return doc

    def aadd_texts(self, doc, texts):
        if doc.dockey in self.docs:
            return False
        self.docs[doc.dockey] = doc
        # 导入片段可能来自其他模型；复制后清除向量，避免混用空间或改动原工作区。
        self.texts.extend(t.model_copy(update={"embedding": None}) for t in texts)
        self._keyword_index = None
        return True

    def _scope(self, session, doc_ids=None, object_ids=None):
        universe = set(self.docs)
        allowed = (
            set(session.spec.allowed_doc_ids) if session.spec.allowed_doc_ids else universe
        )
        allowed &= universe
        if doc_ids is not None:
            requested = set(doc_ids)
            if not requested <= allowed:
                raise ValueError("请求的文档未加载或不在本任务允许范围")
            allowed &= requested
        if object_ids:
            objects = {o.object_id: o for o in session.spec.objects}
            if not set(object_ids) <= set(objects):
                raise ValueError("未知比较对象")
            ranges = [set(objects[x].doc_ids) for x in object_ids]
            if all(ranges):
                allowed &= set.union(*ranges)
        return allowed

    async def _build_texts_index(self, session):
        signature = self._embedding_identity()
        # 模型名和参数共同定义向量空间；配置变化后，旧向量必须重新编码。
        if signature != self._embedding_signature:
            self.index.clear()
            for text in self.texts:
                text.embedding = None
            self._embedding_model = None
            self._embedding_signature = signature
        missing = [t for t in self.texts if t not in self.index]
        if not missing:
            return
        vectors_needed = [t for t in missing if t.embedding is None]
        if vectors_needed:
            session.usage["embedding_requests"] += 1
            if self.settings.embedding != "sparse" and not self.settings.embedding.startswith("local/"):
                session.usage["embedding_tokens_unknown"] += 1
            vectors = await self.embedding_model.embed_documents(
                [t.text for t in vectors_needed]
            )
            for text, vector in zip(vectors_needed, vectors, strict=True):
                text.embedding = vector
        await self.index.add_texts_and_embeddings(missing)

    async def retrieve_texts(self, question, session, doc_ids=None, object_ids=None, k=None):
        selected, _ = await self._retrieve(question, session, doc_ids, object_ids, k)
        return selected

    async def _retrieve(self, question, session, doc_ids, object_ids, k=None):
        # 本层只召回允许范围内的 Text；“片段相关”之后还要经 aget_evidence 提炼事实。
        scope = self._scope(session, doc_ids, object_ids)
        if not scope:
            return [], []
        count = k or self.settings.answer.evidence_k
        remaining, degraded, cache_hit = [], None, False
        if self.settings.answer.retrieval_mode == "hybrid":
            snapshot = self.snapshot
            shared = (
                snapshot is not None and snapshot.keywords is not None
                and all(t.chunk_id in snapshot.keywords.order for t in self.texts)
            )
            store = (
                snapshot.vectors if shared and snapshot.manifest.get("embedding")
                == self._embedding_identity() else None
            )
            if shared:
                keywords = snapshot.keywords
            else:
                if self._keyword_index is None:
                    self._keyword_index = await asyncio.to_thread(ChunkKeywordIndex, self.texts)
                keywords = self._keyword_index
            cache = self.retrieval_cache if store is not None else None
            key = (
                (snapshot.generation, self._embedding_identity(), sorted(scope), question, count)
                if cache else None
            )
            ranked = await cache.get(key, keywords, scope) if cache else None
            cache_hit = ranked is not None
            if ranked is None:
                try:
                    if store is None:
                        await self._build_texts_index(session)
                        store = self.index
                    self._record_query_embedding(session)
                    ranked = await hybrid_candidates(
                        store, keywords, question, count, self.embedding_model, scope
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    degraded = type(exc).__name__
                    lexical = await asyncio.to_thread(keywords.search, question, scope, count * 3)
                    ranked = [(t, 1 / (60 + i)) for i, t in enumerate(lexical, 1)]
                if cache and degraded is None:
                    await cache.put(key, ranked)
            texts = [text for text, _ in ranked[:count]]
            scores = [score for _, score in ranked[:count]]
            remaining = [text for text, _ in ranked[count:]]
        else:
            await self._build_texts_index(session)
            self._record_query_embedding(session)
            texts, scores = await self.index.max_marginal_relevance_search(
                question, count, max(count * 3, count), self.embedding_model, scope
            )
        session.record(
            "retrieval",
            question=question,
            method=self.settings.answer.retrieval_mode,
            doc_ids=sorted(scope),
            chunks=[{"chunk_id": t.chunk_id, "score": s} for t, s in zip(texts, scores)],
            remaining_chunk_ids=[t.chunk_id for t in remaining],
            fallback=degraded,
            cache_hit=cache_hit,
        )
        return texts, remaining

    def _record_query_embedding(self, session):
        session.usage["embedding_requests"] += 1
        if self.settings.embedding != "sparse" and not self.settings.embedding.startswith("local/"):
            session.usage["embedding_tokens_unknown"] += 1

    async def _map_evidence(self, texts, question, spec, session):
        semaphore = asyncio.Semaphore(self.settings.answer.concurrency)
        return await asyncio.gather(
            *(
                map_fxn_summary(
                    t, question, spec, self.settings.get_summary_llm(), session, semaphore
                )
                for t in texts
            ),
            return_exceptions=True,
        )

    def _merge_evidence(self, session, batches, errors, issues):
        for batch in batches:
            if isinstance(batch, BaseException):
                if isinstance(batch, asyncio.CancelledError):
                    raise batch
                errors.append(type(batch).__name__)
                continue
            session.facts, changed = comparison.merge_facts(session.facts, batch.facts)
            batch.context.fact_ids = [
                f.fact_id
                for f in session.facts
                if any(s.chunk_id == batch.context.chunk_id for s in f.sources)
            ]
            session.contexts.append(batch.context)
            issues.extend(batch.issues)
            if changed:
                session.invalidate_answer()

    async def aget_evidence(
        self,
        session,
        evidence_question,
        object_ids=None,
        dimensions=None,
        doc_ids=None,
        chunk_ids=None,
        revise_fact_ids=None,
    ):
        if session.spec is None:
            raise ValueError("任务尚未初始化")
        scope = self._scope(session, doc_ids, object_ids)
        spec = session.spec.model_copy(deep=True)
        if object_ids:
            spec.objects = [o for o in spec.objects if o.object_id in object_ids]
        if dimensions:
            if not set(dimensions) <= set(spec.dimensions):
                raise ValueError("未知比较维度")
            spec.dimensions = dimensions
        remaining = []
        if chunk_ids is not None:
            mapping = {t.chunk_id: t for t in self.texts if t.doc.dockey in scope}
            if not set(chunk_ids) <= set(mapping):
                raise ValueError("指定片段不在任务范围")
            texts = [mapping[c] for c in dict.fromkeys(chunk_ids)]
        else:
            texts, remaining = await self._retrieve(
                evidence_question, session, doc_ids, object_ids
            )
        before = session.facts
        revise_fact_ids = revise_fact_ids or []
        old = {f.fact_id: f for f in session.facts}
        if revise_fact_ids and (not chunk_ids or not set(revise_fact_ids) <= set(old)):
            raise ValueError("纠正旧事实需指定真实 fact_id 及精确 chunk_ids")
        if any(
            not {s.chunk_id for s in old[fid].sources}.intersection(chunk_ids or [])
            for fid in revise_fact_ids
        ):
            raise ValueError("待纠正事实与本轮原文片段没有关联")
        batches = await self._map_evidence(texts, evidence_question, spec, session)
        errors, issues = [], []
        replacements = [
            f
            for batch in batches
            if not isinstance(batch, BaseException)
            for f in batch.facts
            if f.status == "candidate"
        ]
        for fact_id in revise_fact_ids:
            previous = old[fact_id]
            source_ids = {s.chunk_id for s in previous.sources}
            matching = [
                f
                for f in replacements
                if (f.object_id, f.dimension) == (previous.object_id, previous.dimension)
                and source_ids.intersection(s.chunk_id for s in f.sources)
            ]
            if not matching:
                issues.append(f"{fact_id} 未取得同对象同维度的新依据，原候选继续保留")
                continue
            session.facts = [f.model_copy(deep=True) for f in session.facts]
            target = next(f for f in session.facts if f.fact_id == fact_id)
            target.status = "superseded"
            target.issue = "经指定原文重新提炼后由新候选替代；不代表程序已证明真假"
            session.invalidate_answer()
            session.record(
                "fact_revised",
                old_fact_id=fact_id,
                source_chunk_ids=sorted({s.chunk_id for f in matching for s in f.sources}),
            )
        self._merge_evidence(session, batches, errors, issues)
        if remaining and not errors and self.settings.answer.rerank.enabled:
            from paperqa.rerank import supplement

            extra, focused_question, focused_spec = await supplement(
                session, spec, evidence_question, remaining, self.settings.answer.rerank
            )
            if extra:
                added = await self._map_evidence(extra, focused_question, focused_spec, session)
                self._merge_evidence(session, added, errors, issues)
                texts.extend(extra)
        result = comparison.summarize_progress(session.spec, before, session.facts)
        result.update(
            {"errors": errors, "issues": issues, "chunk_ids": [t.chunk_id for t in texts]}
        )
        session.record("evidence_ready", **result)
        return result

    def read_source(self, session, chunk_id, neighbors=1, object_ids=None):
        if not 0 <= neighbors <= 3:
            raise ValueError("相邻范围应在0到3之间")
        scope = self._scope(session, object_ids=object_ids)
        mapping = {t.chunk_id: t for t in self.texts if t.doc.dockey in scope}
        if chunk_id not in mapping:
            raise ValueError("片段不在当前已加载的允许范围")
        target = mapping[chunk_id]
        same_doc = sorted(
            [t for t in mapping.values() if t.doc.dockey == target.doc.dockey],
            key=lambda t: t.order,
        )
        position = same_doc.index(target)
        chosen = same_doc[max(0, position - neighbors) : position + neighbors + 1]
        # 先放目标，再放邻接；这样大邻接块不会挤掉用户指定片段。
        chosen = [target] + [t for t in chosen if t is not target]
        pieces, used, included, pages = [], 0, [], set()
        included_texts, skipped_neighbors = [], []
        limit = self.settings.answer.source_read_chars
        truncated = False
        for text in chosen:
            if text is not target and text.layout_uncertain and not target.layout_uncertain:
                skipped_neighbors.append(text.chunk_id)
                continue
            piece = (
                f"[chunk_id={text.chunk_id}; PDF页={text.pages}; kind={text.kind}]\n{text.text}"
            )
            if text.layout_uncertain:
                piece = "[该页表头列关系未恢复，不能据此确认数值归属]\n" + piece
            if used + len(piece) > limit:
                if text is target:
                    truncated = True
                    piece = piece[:limit] + "\n[已截短，不能据此确认完整表格]"
                else:
                    skipped_neighbors.append(text.chunk_id)
                    continue
            pieces.append(piece)
            included.append(text.chunk_id)
            included_texts.append(text)
            pages.update(text.pages)
            used += len(piece)
        content = "\n\n".join(pieces)
        read = SourceRead(
            read_id=stable_id(content),
            doc_id=target.doc.dockey,
            chunk_ids=included,
            content=content,
            pages=sorted(pages),
            truncated=truncated,
            layout_uncertain=any(t.layout_uncertain for t in included_texts),
            targets=object_ids or [],
        )
        if not any(
            r.read_id == read.read_id and r.targets == read.targets
            for r in session.source_reads
        ):
            session.source_reads.append(read)
            session.invalidate_answer()
        session.record("source_read", **read.model_dump(), skipped_neighbor_ids=skipped_neighbors)
        return read

    async def aquery(self, session):
        session.invalidate_answer()
        # 事实池中的内容未必全部装得进回答；先按预算选材料，再让模型组织解释。
        selection = comparison.select_contexts(
            session.spec,
            session.facts,
            session.contexts,
            session.source_reads,
            self.settings.answer.context_chars,
            self.settings.answer.max_sources,
            session.question,
        )
        data = comparison.build_answer_context(session.spec, selection, session.question)
        if selection["budget_too_small"]:
            explanation = "上下文预算不足以容纳任务与来源元数据，请缩小比较范围或增加预算。"
        elif not data["facts"] and not any(
            not r["layout_uncertain"] for r in data["source_reads"]
        ):
            explanation = "当前没有可用于回答的可靠候选证据；若表格列关系未恢复，请通过原页核对。未取得依据不等于论文未报告。"
        else:
            response = await request_model(
                session,
                self.settings.get_llm(),
                [
                    {"role": "system", "content": prompts.ANSWER},
                    {"role": "user", "content": dump_json(data)},
                ],
                role="answer",
            )
            explanation = response.choices[0].message.content or ""
            if not explanation.strip():
                raise ValueError("回答模型没有返回解释")
            allowed = (
                {f["fact_id"] for f in data["facts"]}
                | {c for r in data["source_reads"] for c in r["chunk_ids"]}
                | {s["chunk_id"] for f in data["facts"] for s in f["sources"]}
            )
            claimed = set(
                re.findall(
                    r"\[(?:fact_id=|chunk_id=|citation:)?(f_[a-f0-9]+|[a-f0-9]{24})\]",
                    explanation,
                )
            )
            if not claimed <= allowed:
                raise ValueError("解释引用了本次回答不存在的来源标识")
        session.answer_data = data
        session.explanation = explanation
        session.answer = (
            (
                comparison.render_table(data) + "\n\n"
                if session.spec.mode == "comparison"
                else ""
            )
            + explanation
            + "\n\n"
            + ("依据摘录\n\n" + comparison.render_sources(data))
        )
        session.answer_is_current = True
        session.record(
            "answer_ready",
            fact_ids=[f["fact_id"] for f in data["facts"]],
            coverage=data["coverage"],
            excluded_fact_ids=data["excluded_fact_ids"],
        )
        return session
