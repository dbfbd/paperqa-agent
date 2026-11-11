"""单片段提炼及统一模型用量记录；不搜索资料库、不改写原问题。"""

import asyncio
import json
import re
import time

from paperqa import prompts
from paperqa.types import Context, EvidenceBatch, EvidenceFact, SourceRef
from paperqa.utils import dump_json, normalized


class BudgetExceeded(RuntimeError):
    pass


async def request_model(session, model, messages, role="model", **kwargs):
    used = session.usage["prompt_tokens"] + session.usage["completion_tokens"]
    if session.usage["requests"] >= session.max_calls or used >= session.max_tokens:
        raise BudgetExceeded("本次模型调用次数或已知 token 预算耗尽")
    if role == "evidence" and (
        session.usage["requests"] >= session.max_calls - 1 or used >= session.max_tokens * 0.75
    ):
        raise BudgetExceeded("停止追加提炼，为当前答案预留调用和 token 预算")
    # 检查与加计数之间没有 await，片段并发时也不会同时占用同一个名额。
    session.usage["requests"] += 1
    start = time.monotonic()
    try:
        response = await model.get_router().acompletion(
            model=model.name, messages=messages, **kwargs
        )
    except BaseException as exc:
        session.usage["failed_requests"] += 1
        session.usage["missing_usage_requests"] += 1
        session.record(
            "model_request",
            role=role,
            model=model.name,
            ok=False,
            error=type(exc).__name__,
            seconds=time.monotonic() - start,
        )
        raise
    usage = getattr(response, "usage", None)
    if usage is None:
        session.usage["missing_usage_requests"] += 1
    else:
        session.add_tokens(usage.prompt_tokens or 0, usage.completion_tokens or 0)
    session.record(
        "model_request",
        role=role,
        model=model.name,
        ok=True,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        seconds=time.monotonic() - start,
    )
    return response


def llm_parse_json(content: str):
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise TypeError("预期 JSON 对象")
    return payload


async def _map_fxn_summary(text, question, spec, model, session):
    messages = [
        {"role": "system", "content": prompts.EVIDENCE},
        {
            "role": "user",
            "content": dump_json(
                {
                    "local_question": question,
                    "targets": spec.model_dump(),
                    "source": {
                        "chunk_id": text.chunk_id,
                        "doc_id": text.doc.dockey,
                        "pages": text.pages,
                        "kind": text.kind,
                        "layout_uncertain": text.layout_uncertain,
                        "text": text.text,
                    },
                }
            ),
        },
    ]
    objects = {o.object_id: o for o in spec.objects}
    for attempt in range(2):
        response = await request_model(session, model, messages, role="evidence")
        raw = response.choices[0].message.content or ""
        try:
            data = llm_parse_json(raw)
            if not isinstance(data.get("facts", []), list):
                raise TypeError("facts 必须是列表")
            facts, issues = [], [str(x) for x in data.get("issues", [])]
            for item in data.get("facts", []):
                object_id, dim = item["object_id"], item["dimension"]
                if object_id not in objects or dim not in spec.dimensions:
                    issues.append("候选对象或维度不在任务中，已忽略")
                    continue
                obj = objects[object_id]
                if obj.doc_ids and text.doc.dockey not in obj.doc_ids:
                    issues.append(f"{object_id} 的候选来源超出关联文档，已忽略")
                    continue
                quote = str(item.get("quote", ""))
                # 摘录可定位只是最低来源检查，不能证明模型理解了表头和实验条件。
                supported = bool(normalized(quote)) and normalized(quote) in normalized(
                    text.text
                )
                ambiguous_result = (
                    text.layout_uncertain
                    and re.search(
                        r"result|score|\bEM\b|F1|accuracy|结果|成绩|得分", dim, re.IGNORECASE
                    )
                    and re.search(r"\d", str(item["value"]))
                )
                issue = "" if supported else "摘录无法在当前原文定位"
                value = str(item["value"])
                if re.fullmatch(r"[+-]?\d+(?:\.\d+)?[%‰]?", value.strip()) and normalized(
                    value
                ) not in normalized(quote):
                    supported = False
                    issue = "候选数值未出现在依据摘录中"
                    issues.append(issue)
                if ambiguous_result:
                    supported = False
                    issue = "所在页的表头列关系未恢复，数值归属暂不作为有效依据"
                    issues.append(issue)
                sources = (
                    [
                        SourceRef(
                            doc_id=text.doc.dockey,
                            chunk_id=text.chunk_id,
                            pages=text.pages,
                            quote=quote,
                        )
                    ]
                    if quote.strip()
                    else []
                )
                facts.append(
                    EvidenceFact(
                        object_id=object_id,
                        dimension=dim,
                        value=str(item["value"]),
                        unit=str(item.get("unit", "")),
                        conditions=item.get("conditions") or {},
                        sources=sources,
                        status="candidate" if supported else "unsupported",
                        issue=issue,
                    )
                )
            return EvidenceBatch(
                context=Context(
                    chunk_id=text.chunk_id,
                    question=question,
                    summary=str(data.get("summary", "")),
                    score=max(0, min(10, int(data.get("score", 0)))),
                    fact_ids=[f.fact_id for f in facts],
                ),
                facts=facts,
                issues=issues,
            )
        except (ValueError, TypeError, KeyError) as exc:
            if attempt:
                raise ValueError("片段提炼的结构化结果仍无效") from exc
            messages += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": "格式校验失败，请按规定 JSON 格式重发，不增加原文之外事实。",
                },
            ]


async def map_fxn_summary(text, question, spec, model, session, semaphore=None):
    semaphore = semaphore or asyncio.Semaphore(1)
    async with semaphore:
        return await _map_fxn_summary(text, question, spec, model, session)
