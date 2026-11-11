"""在证据缺口出现后，为一个对象和指标重排尚未取证的候选。"""

import asyncio
import math
import re
from functools import lru_cache
from threading import Lock

from paperqa.comparison import compute_coverage
from paperqa.retrieval import lexical_tokens

_worker = Lock()
_generic = {"result", "answer", "dataset", "split", "metric", "extra_training_data"}


def focus_gap(spec, facts, remaining, examined, limit):
    gaps = sorted(
        (c for c in compute_coverage(spec, facts) if c["status"] == "missing"),
        key=lambda c: c["dimension"] in _generic - {"result", "answer"},
    )
    objects = {obj.object_id: obj for obj in spec.objects}
    for gap in gaps:
        obj, dimension = objects[gap["object_id"]], gap["dimension"]
        clues = {x.casefold() for x in lexical_tokens(
            f"{obj.name} {obj.experiment_hint} {dimension}"
        ) if len(x) > 1 and x.casefold() not in _generic}
        choices = []
        for text in remaining:
            if text.chunk_id in examined:
                continue
            if obj.doc_ids:
                if text.doc.dockey not in obj.doc_ids:
                    continue
            elif not any(
                re.search(r"(?<![A-Za-z0-9])" + re.escape(clue) + r"(?![A-Za-z0-9])",
                          text.text, re.IGNORECASE)
                for clue in clues
            ):
                continue
            choices.append(text)
        if choices:
            return obj, dimension, choices[:limit]
    return None


@lru_cache(maxsize=2)
def _load(path, revision, offline):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        path, revision=revision, local_files_only=offline, max_length=512, device="cpu",
    )


async def rerank(question, candidates, config):
    def score():
        # wait_for 取消的是等待者，工作线程继续完成。单工作位避免超时后推理堆积。
        if not _worker.acquire(blocking=False):
            raise RuntimeError("重排模型正忙")
        try:
            model = _load(config.path or config.model, config.revision, config.local_files_only)
            scores = model.predict(
                [(question, t.text) for t in candidates], batch_size=8, show_progress_bar=False
            )
            scores = [float(s) for s in scores]
            if len(scores) != len(candidates) or not all(map(math.isfinite, scores)):
                raise ValueError("重排分数无效")
            return [t for t, _ in sorted(
                zip(candidates, scores, strict=True), key=lambda row: -row[1]
            )]
        finally:
            _worker.release()

    return await asyncio.wait_for(asyncio.to_thread(score), config.timeout)


async def supplement(session, spec, question, remaining, config):
    if any(event["event"] == "rerank_attempt" for event in session.events):
        return [], question, spec
    used = session.usage["prompt_tokens"] + session.usage["completion_tokens"]
    budget = min(config.evidence, session.max_calls - session.usage["requests"] - 1)
    if budget <= 0 or used >= session.max_tokens * 0.75:
        return [], question, spec
    target = focus_gap(
        spec, session.facts, remaining, {c.chunk_id for c in session.contexts}, config.candidates
    )
    if target is None:
        return [], question, spec
    obj, dimension, candidates = target
    focused = f"{question}\n{obj.name} · {dimension} · {obj.experiment_hint}"
    session.record(
        "rerank_attempt", object_id=obj.object_id, dimension=dimension,
        candidate_ids=[t.chunk_id for t in candidates],
    )
    try:
        ranked = (await rerank(focused, candidates, config))[:budget]
    except Exception as exc:  # noqa: BLE001 -- 可选补查失败时保留首轮证据。
        session.record("rerank_fallback", error=type(exc).__name__)
        return [], question, spec
    session.record("rerank_selected", chunk_ids=[t.chunk_id for t in ranked])
    focused_spec = spec.model_copy(update={"objects": [obj], "dimensions": [dimension]})
    return ranked, focused, focused_spec
