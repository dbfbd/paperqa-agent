"""保留一个 Aviary ToolSelector 运行入口，业务能力在下层模块中。"""

import asyncio
import time

from aviary.core import Message, ToolSelector, ToolSelectorLedger
from aviary.message import MalformedMessageError

from paperqa import comparison, prompts
from paperqa.agents.env import PaperQAEnvironment
from paperqa.agents.models import AgentStatus, AnswerResponse
from paperqa.core import BudgetExceeded, llm_parse_json, request_model
from paperqa.docs import Docs
from paperqa.settings import Settings
from paperqa.types import ComparisonSpec, PQASession
from paperqa.utils import dump_json


async def prepare_task(session, settings, catalog, explicit_spec=None, allowed_doc_ids=None):
    known = {d["doc_id"] for d in catalog}
    allowed = known if allowed_doc_ids is None else set(allowed_doc_ids)
    if not allowed <= known:
        raise ValueError("任务指定了未知文档")
    if explicit_spec is None:
        response = await request_model(
            session,
            settings.get_llm(),
            [
                {"role": "system", "content": prompts.TASK},
                {
                    "role": "user",
                    "content": dump_json(
                        {
                            "question": session.question,
                            "documents": [d for d in catalog if d["doc_id"] in allowed],
                        }
                    ),
                },
            ],
            role="task",
        )
        spec = ComparisonSpec.model_validate(
            llm_parse_json(response.choices[0].message.content or "")
        )
    else:
        spec = (
            explicit_spec
            if isinstance(explicit_spec, ComparisonSpec)
            else ComparisonSpec.model_validate(explicit_spec)
        )
    requested = set(spec.allowed_doc_ids) if spec.allowed_doc_ids else allowed
    if not requested <= allowed or any(not set(o.doc_ids) <= requested for o in spec.objects):
        raise ValueError("比较要求含范围外的文档标识")
    # 空列表在工具层表示不限制；允许集合为空时，入口应直接报材料不足。
    if not requested:
        raise ValueError("没有允许使用的材料")
    session.spec = spec.model_copy(deep=True, update={"allowed_doc_ids": sorted(requested)})
    session.record("task_prepared", spec=session.spec.model_dump())


async def run_agent(env, settings):
    session = env.state.session
    model = settings.get_agent_llm()

    async def completion(_model, **kwargs):
        return await request_model(session, model, role="decision", **kwargs)

    selector = ToolSelector(model_name=model.name, acompletion=completion)
    _, tools = await env.reset()
    ledger = ToolSelectorLedger(
        messages=[Message(role="system", content=prompts.AGENT)], tools=tools
    )
    turns = []
    for _ in range(settings.agent.max_steps):
        used = session.usage["prompt_tokens"] + session.usage["completion_tokens"]
        if (
            (session.facts or session.source_reads)
            and not session.answer_is_current
            and (
                used >= session.max_tokens * 0.75
                or session.usage["requests"] >= session.max_calls - 2
            )
        ):
            session.record("answer_reserve", used_tokens=used, reason="保留预算生成当前答案")
            await env.state.docs.aquery(session)
            session.stop_reason = "answer_reserve"
            return AgentStatus.PARTIAL
        overview = {
            "question": session.question,
            "task": session.spec.model_dump(),
            "documents": [
                {"doc_id": d.dockey, "name": d.docname} for d in env.state.docs.docs.values()
            ],
            **env.state.status(),
            "read_chunk_ids": sorted({c for r in session.source_reads for c in r.chunk_ids}),
        }
        # 每轮保留当前状态和最近两轮反馈，避免完整工具历史不断挤占决策上下文。
        ledger.messages = [
            Message(role="system", content=prompts.AGENT),
            Message(content=dump_json(overview)),
            *[message for turn in turns[-2:] for message in turn],
        ]
        session.record(
            "decision_context",
            message_count=len(ledger.messages),
            content_chars=sum(len(m.content or "") for m in ledger.messages),
        )
        for attempt in range(2):
            try:
                action = await selector(ledger.messages, tools)
                break
            except MalformedMessageError:
                if attempt:
                    raise
        obs, _, done, _ = await env.step(action)
        turns.append([action, *obs])
        if done:
            gaps = any(c["status"] != "found" for c in session.answer_data.get("coverage", []))
            return AgentStatus.PARTIAL if gaps else AgentStatus.COMPLETED
    if (session.facts or session.source_reads) and not session.answer_is_current:
        session.record("answer_reserve", reason="轮次用尽，尝试生成当前答案")
        await env.state.docs.aquery(session)
    session.stop_reason = "step_budget"
    return AgentStatus.BUDGET


async def _run_with_timeout_failure(operation, session, docs, settings):
    try:
        async with asyncio.timeout(settings.agent.timeout):
            return await operation()
    except TimeoutError:
        session.stop_reason = "timeout"
        status = AgentStatus.TIMEOUT
    except BudgetExceeded:
        session.stop_reason = "call_or_token_budget"
        status = AgentStatus.BUDGET
    except Exception as exc:  # noqa: BLE001 -- 运行边界必须保留失败及已有证据。
        session.record("run_error", error=type(exc).__name__, issue=str(exc)[:400])
        session.stop_reason = "error"
        status = AgentStatus.ERROR
    # 有效答案可保留为部分结果；旧答案不会因异常兜底重新变有效。
    return status


async def agent_query(
    question, settings=None, docs=None, spec=None, index=None, allowed_doc_ids=None,
    *, retrieval_cache=None,
):
    settings = settings or Settings()
    docs = docs or Docs(settings)
    docs.snapshot = index
    docs.retrieval_cache = retrieval_cache
    # Docs 管已加载材料与检索索引；PQASession 记录这次任务的事实、答案和预算。
    session = PQASession(
        question=question,
        max_calls=settings.agent.max_calls,
        max_tokens=settings.agent.max_tokens,
    )
    start = time.monotonic()
    if index:
        session.record("index_bound", version=index.generation)

    async def operation():
        catalog = (
            index.document_list()
            if index
            else [{"doc_id": d.dockey, "name": d.docname} for d in docs.docs.values()]
        )
        await prepare_task(session, settings, catalog, spec, allowed_doc_ids)
        env = PaperQAEnvironment(docs, session, settings, index)
        return await run_agent(env, settings)

    status = await _run_with_timeout_failure(operation, session, docs, settings)
    if not session.answer_is_current and session.spec:
        selection = comparison.select_contexts(
            session.spec,
            session.facts,
            session.contexts,
            session.source_reads,
            settings.answer.context_chars,
            settings.answer.max_sources,
            question,
        )
        session.answer_data = comparison.build_answer_context(session.spec, selection, question)
    return AnswerResponse.from_session(session, docs, status, time.monotonic() - start)
