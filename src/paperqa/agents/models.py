"""运行状态不是答案质量评分；输出为独立快照。"""

from copy import deepcopy
from enum import StrEnum
from typing import Any

from pydantic import Field

from paperqa.types import Model


class AgentStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    BUDGET = "budget_exhausted"
    TIMEOUT = "timeout"
    ERROR = "error"


class AnswerResponse(Model):
    run_id: str
    question: str
    status: AgentStatus
    answer: str
    explanation: str = ""
    answer_is_current: bool
    result: dict[str, Any] = Field(default_factory=dict)
    documents: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    stop_reason: str | None = None
    elapsed_seconds: float = 0
    tool_history: list[list[str]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    quality_verified: bool = False

    @classmethod
    def from_session(cls, session, docs, status, elapsed):
        return cls(
            run_id=session.run_id,
            question=session.question,
            status=status,
            answer=session.answer
            if session.answer_is_current
            else "运行未完成，以下结果仅为当前证据快照。",
            explanation=session.explanation if session.answer_is_current else "",
            answer_is_current=session.answer_is_current,
            result=deepcopy(session.answer_data),
            documents=[d.model_dump() for d in docs.docs.values()],
            usage=dict(session.usage),
            stop_reason=session.stop_reason,
            elapsed_seconds=elapsed,
            tool_history=deepcopy(session.tool_history),
            trace=deepcopy(session.events),
        )
