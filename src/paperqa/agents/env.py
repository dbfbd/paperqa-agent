"""Aviary 工具装配和串行分派；取证内部仍可并发。"""

import json

import jsonschema
from aviary.core import Environment, Message, Tool, ToolRequestMessage, ToolResponseMessage

from paperqa.agents.tools import (
    Complete,
    EnvironmentState,
    GatherEvidence,
    GenerateAnswer,
    PaperSearch,
    ReadSource,
)
from paperqa.types import PQASession
from paperqa.utils import dump_json


def settings_to_tools(settings, index=None):
    methods = [
        PaperSearch(settings, index).paper_search,
        GatherEvidence(settings).gather_evidence,
        ReadSource(settings).read_source,
        GenerateAnswer(settings).gen_answer,
        Complete(settings).complete,
    ]
    tools = [Tool.from_function(fn, concurrency_safe=False) for fn in methods]
    for tool in tools:
        tool.info.parameters.additionalProperties = False
    return tools


def decision_response(message):
    """只压缩发给决策模型的副本，完整回执继续保存在审计轨迹中。"""
    try:
        data = json.loads(message.content)
    except (TypeError, ValueError):
        return message
    if not isinstance(data, dict):
        return message
    for name in ("errors", "issues"):
        if name in data:
            data[name] = list(dict.fromkeys(data[name]))[:8]
    for fact in data.get("recent_facts", []):
        for source in fact.get("sources", []):
            source.pop("quote", None)
    if "answer" in data:
        data["answer_preview"] = data.pop("answer")[:600]
    return message.model_copy(update={"content": dump_json(data)})


class PaperQAEnvironment(Environment):
    def __init__(self, docs, session, settings, index=None):
        self.settings = settings
        self.state = self.make_initial_state(docs, session)
        self.tools = settings_to_tools(settings, index)
        self._has_reset = False

    @staticmethod
    def make_initial_state(docs, session):
        # session 必须由每次入口新建；保留外部明确预载的原文。
        return EnvironmentState(docs, session)

    async def reset(self):
        if self._has_reset:
            session = self.state.session
            fresh = PQASession(
                question=session.question,
                spec=session.spec.model_copy(deep=True),
                max_calls=session.max_calls,
                max_tokens=session.max_tokens,
            )
            for name in PQASession.model_fields:
                setattr(session, name, getattr(fresh, name))
        self._has_reset = True
        self.state.session.record("environment_reset", **self.state.status())
        return [
            Message(
                content=dump_json(
                    {
                        "question": self.state.session.question,
                        "task": self.state.session.spec.model_dump(),
                        **self.state.status(),
                    }
                )
            )
        ], self.tools

    async def step(self, action):
        session = self.state.session
        if not isinstance(action, ToolRequestMessage) or not action.tool_calls:
            return [Message(content="请调用工具继续或申请结束。")], 0.0, False, False
        session.tool_history.append([c.function.name for c in action.tool_calls])
        responses = []
        for call in action.tool_calls:
            session.record(
                "agent_action", name=call.function.name, arguments=call.function.arguments
            )
            if session.stop_reason is not None:
                responses.append(
                    ToolResponseMessage.from_call(call, content="结束已接受，本工具未执行。")
                )
                continue
            try:
                tool = next((t for t in self.tools if t.info.name == call.function.name), None)
                if tool is None:
                    raise ValueError("未知工具名称")
                # 模型只提出工具调用；参数校验通过后，才注入真实任务状态执行。
                jsonschema.validate(call.function.arguments, tool.info.parameters.model_dump())
                result = await self.exec_tool_calls(
                    ToolRequestMessage(tool_calls=[call]),
                    concurrency=False,
                    state=self.state,
                    handle_tool_exc=True,
                )
                responses.extend(result)
            except (ValueError, jsonschema.ValidationError) as exc:
                responses.append(
                    ToolResponseMessage.from_call(
                        call,
                        content=dump_json(
                            {"error": type(exc).__name__, "issue": str(exc)[:400]}
                        ),
                    )
                )
        session.record(
            "tool_responses", responses=[r.model_dump(mode="json") for r in responses]
        )
        session.record("after_step", **self.state.status())
        return (
            [decision_response(r) for r in responses],
            0.0,
            session.stop_reason is not None,
            False,
        )
