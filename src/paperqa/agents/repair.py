"""作答前按证据覆盖状态补查一个缺项，复用当前索引快照。"""

import asyncio
import re

from paperqa.comparison import compute_coverage


def named_tables(texts, scope, hint):
    numbers = set(re.findall(
        r"(?:(?<![A-Za-z])table\s+|表\s*)(\d+)(?![0-9A-Za-z]|\.\d)", hint, re.IGNORECASE,
    ))
    if len(numbers) != 1:
        return []
    caption = re.compile(
        rf"^[ \t]*(?:table[ \t]+|表[ \t]*){numbers.pop()}[ \t]*(?:[:：]|\.(?!\d))",
        re.IGNORECASE | re.MULTILINE,
    )
    return [
        text for text in texts
        if text.doc.dockey in scope and text.kind == "table"
        and not text.layout_uncertain and caption.search(text.text.partition("[同页正文]")[0])
    ]


async def repair_missing_evidence(docs, session):
    spec = session.spec
    used = session.usage["prompt_tokens"] + session.usage["completion_tokens"]
    budget = min(2, session.max_calls - session.usage["requests"] - 1)
    if (
        spec.mode != "comparison"
        or not (session.contexts or session.source_reads or session.facts)
        or budget <= 0
        or used >= session.max_tokens * 0.75
        or any(e["event"] == "gap_repair_started" for e in session.events)
    ):
        return None

    gaps = [row for row in compute_coverage(spec, session.facts) if row["status"] == "missing"]
    if not gaps:
        return None
    objects = {obj.object_id: obj for obj in spec.objects}
    gap = min(gaps, key=lambda row: (
        row["dimension"] in {"dataset", "split", "metric", "extra_training_data"},
        not (set(objects[row["object_id"]].doc_ids) - docs.docs.keys()),
    ))
    target, dimension = objects[gap["object_id"]], gap["dimension"]
    question = f"{target.name} {dimension} {target.experiment_hint or session.question[:400]}"
    session.record(
        "gap_repair_started", object_id=target.object_id, dimension=dimension, query=question,
    )

    snapshot = docs.snapshot
    available = set(docs.docs) | (snapshot.doc_ids if snapshot is not None else set())
    allowed = available & set(spec.allowed_doc_ids) if spec.allowed_doc_ids else available
    selected = sorted(allowed & set(target.doc_ids))
    opened = []
    try:
        if not target.doc_ids and snapshot is not None:
            hits = await snapshot.discover(question, 2, allowed & snapshot.doc_ids)
            selected = [hit["doc_id"] for hit in hits]
        if not selected and not target.doc_ids:
            selected = sorted(allowed & docs.docs.keys())
        if snapshot is not None:
            for doc_id in [key for key in selected if key not in docs.docs][:2]:
                doc, texts = await asyncio.to_thread(snapshot.get_saved_object, doc_id)
                if docs.aadd_texts(doc, texts):
                    opened.append(doc_id)
        scope = [doc_id for doc_id in selected if doc_id in docs.docs]
        examined = {context.chunk_id for context in session.contexts}
        candidates = [
            text for text in named_tables(docs.texts, set(scope), target.experiment_hint or question)
            if text.chunk_id not in examined
        ]
        strategy = "table_caption" if candidates else "focused_query"
        if not candidates and scope:
            candidates = await docs.retrieve_texts(question, session, scope, [target.object_id], k=6)
        chunks = [text.chunk_id for text in candidates if text.chunk_id not in examined][:budget]
        if chunks:
            await docs.aget_evidence(
                session, question, [target.object_id], [dimension], scope, chunk_ids=chunks,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        result = {"status": "error", "error": type(exc).__name__, "opened_doc_ids": opened}
    else:
        status = next(
            row["status"] for row in compute_coverage(spec, session.facts)
            if row["object_id"] == target.object_id and row["dimension"] == dimension
        )
        result = {
            "status": status, "opened_doc_ids": opened, "chunk_ids": chunks, "strategy": strategy,
        }
        if status == "missing" and strategy == "table_caption" and chunks:
            # 摘录校验失败时保留整表给答案模型回查，事实格继续标为缺失。
            read = docs.read_source(session, chunks[0], neighbors=0, object_ids=[target.object_id])
            result["source_read_id"] = read.read_id
    session.record("gap_repair_finished", object_id=target.object_id, dimension=dimension, **result)
    return result
