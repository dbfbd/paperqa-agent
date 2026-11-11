"""五种面向模型的工具；实例存配置，state 由环境注入。"""

import asyncio

from paperqa.agents.repair import repair_missing_evidence
from paperqa.comparison import compute_coverage


class EnvironmentState:
    def __init__(self, docs, session):
        self.docs = docs
        self.session = session

    def status(self):
        return {
            "paper_count": len(self.docs.docs),
            "text_count": len(self.docs.texts),
            "fact_count": len(self.session.facts),
            "answer_is_current": self.session.answer_is_current,
            "coverage": compute_coverage(self.session.spec, self.session.facts),
        }


class NamedTool:
    def __init__(self, settings):
        self.settings = settings


class PaperSearch(NamedTool):
    def __init__(self, settings, index):
        super().__init__(settings)
        self.index = index

    async def paper_search(self, query: str, doc_ids: list[str] | None = None, state=None):
        """在本地资料库发现并加载文档；仅加载，不提炼实验事实。

        Args:
            query: 搜索关键词，可中英混用；避免全文长问题及复杂查询符号。
            doc_ids: 可选文档限制，必须在本任务允许的文档范围内。
        """
        if self.index is None:
            return {"error": "当前仅有预载材料，没有资料库索引", **state.status()}
        allowed = set(state.session.spec.allowed_doc_ids) or self.index.doc_ids
        if doc_ids is not None:
            if not set(doc_ids) <= allowed:
                raise ValueError("文档范围超出任务允许范围")
            allowed &= set(doc_ids)
        hits = await self.index.discover(query, self.settings.agent.search_count, allowed)
        state.session.record("document_search", query=query, allowed=sorted(allowed), hits=hits)
        result = []
        for hit in hits:
            doc, texts = await asyncio.to_thread(self.index.get_saved_object, hit["doc_id"])
            added = state.docs.aadd_texts(doc, texts)
            result.append(
                {
                    "doc_id": doc.dockey,
                    "name": doc.docname,
                    "added": added,
                    "chunk_count": len(texts),
                    "score": hit["score"],
                    "parse_warnings": doc.parse_warnings,
                }
            )
        return {"documents": result, **state.status()}


class GatherEvidence(NamedTool):
    async def gather_evidence(
        self,
        question: str,
        object_ids: list[str] | None = None,
        dimensions: list[str] | None = None,
        doc_ids: list[str] | None = None,
        chunk_ids: list[str] | None = None,
        revise_fact_ids: list[str] | None = None,
        state=None,
    ):
        """对已加载材料进行局部取证，返回新事实、缺口及冲突。

        Args:
            question: 本轮取证问题，例如只查某实验的额外训练数据，不改变原问题。
            object_ids: 可选任务对象 ID 列表。
            dimensions: 可选比较维度，使用任务中的原维度名。
            doc_ids: 可选已加载文档范围。
            chunk_ids: 可选精确片段范围，用于回查后重新提炼，绕过相似度检索。
            revise_fact_ids: 回查后需纠正的旧事实ID；必须同时指定相关 chunk_ids，旧候选保留历史标记。
        """
        result = await state.docs.aget_evidence(
            state.session, question, object_ids, dimensions, doc_ids, chunk_ids, revise_fact_ids
        )
        valid = [
            f for f in state.session.facts if f.status not in {"unsupported", "superseded"}
        ]
        result["recent_facts"] = [f.model_dump() for f in valid[-8:]]
        result["next_read_candidates"] = [
            {"chunk_id": t.chunk_id, "name": t.name, "kind": t.kind}
            for t in state.docs.texts
            if t.chunk_id in result["chunk_ids"]
        ]
        return result


class ReadSource(NamedTool):
    async def read_source(
        self, chunk_id: str, neighbors: int = 1, object_ids: list[str] | None = None, state=None
    ):
        """回查原文与相邻内容，不调用摘要模型；读过不等于已经核验正确。

        Args:
            chunk_id: 取证反馈给出的真实片段 ID，不是文件路径。
            neighbors: 同文档前后各读取几个相邻片段，0到3；目标片段优先。
            object_ids: 可选本次回查针对的对象 ID。
        """
        return state.docs.read_source(
            state.session, chunk_id, neighbors, object_ids
        ).model_dump()


class GenerateAnswer(NamedTool):
    async def gen_answer(self, state=None):
        """生成有来源的对照或解释。启用补查时先尝试弥补一个证据缺项。"""
        repair = None
        if self.settings.agent.gap_repair:
            repair = await repair_missing_evidence(state.docs, state.session)
        await state.docs.aquery(state.session)
        return {
            "answer": state.session.answer,
            "coverage": state.session.answer_data["coverage"],
            "answer_is_current": True,
            "excluded_fact_ids": state.session.answer_data["excluded_fact_ids"],
            "excluded_source_read_ids": state.session.answer_data["excluded_source_read_ids"],
            **({"gap_repair": repair} if repair is not None else {}),
        }


class Complete(NamedTool):
    async def complete(self, reason: str, state=None):
        """检查当前答案并申请结束；被拒绝时环境继续运行。

        Args:
            reason: enough_evidence、no_progress、cannot_answer、budget 四者之一。
        """
        session = state.session
        if reason not in {"enough_evidence", "no_progress", "cannot_answer", "budget"}:
            return {"accepted": False, "issue": "未知结束原因"}
        if not session.answer or not session.answer_is_current:
            return {"accepted": False, "issue": "没有对应当前证据的答案，请先 gen_answer"}
        session.stop_reason = reason
        return {
            "accepted": True,
            "reason": reason,
            "coverage": session.answer_data["coverage"],
            "quality_verified": False,
        }
