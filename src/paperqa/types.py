"""任务数据及原解析插件契约；这里不导入业务运行层。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChunkMetadata(Model):
    size: int
    overlap: int
    name: str | None = None


class ParsedMetadata(Model):
    parsing_libraries: list[str]
    total_parsed_text_length: int = Field(ge=0)
    count_parsed_media: int = Field(default=0, ge=0)
    paperqa_version: str = "5.26.0+refactor.1"
    name: str | None = None
    chunk_metadata: ChunkMetadata | None = None


class ParsedMedia(Model):
    index: int
    data: bytes = b""
    url: str | None = None
    text: str | None = None
    info: dict[str, Any] = Field(default_factory=dict)


class ParsedText(Model):
    content: str | list[str] | dict[str, str | tuple[str, list[ParsedMedia]]]
    metadata: ParsedMetadata

    def reduce_content(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return "\n\n".join(self.content)
        return "\n\n".join(v if isinstance(v, str) else v[0] for v in self.content.values())


class Doc(Model):
    dockey: str
    docname: str
    citation: str = ""
    source_path: str = ""
    content_hash: str = ""
    parse_warnings: list[str] = Field(default_factory=list)


class Text(Model):
    chunk_id: str
    text: str
    name: str
    doc: Doc
    order: int = 0
    pages: list[int] = Field(default_factory=list)
    kind: Literal["body", "table"] = "body"
    layout_uncertain: bool = False
    embedding: list[float] | None = None

    def __hash__(self) -> int:
        return hash(self.chunk_id)


class SourceRef(Model):
    doc_id: str
    chunk_id: str
    pages: list[int] = Field(default_factory=list)
    quote: str = Field(min_length=1)


class ComparisonObject(Model):
    object_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    doc_ids: list[str] = Field(default_factory=list)
    experiment_hint: str = ""


class ComparisonSpec(Model):
    mode: Literal["qa", "comparison"] = "comparison"
    objects: list[ComparisonObject] = Field(min_length=1)
    dimensions: list[str] = Field(
        default_factory=lambda: [
            "dataset",
            "split",
            "metric",
            "extra_training_data",
            "result",
        ],
        min_length=1,
    )
    allowed_doc_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [o.object_id for o in self.objects]
        if len(ids) != len(set(ids)) or len(self.dimensions) != len(set(self.dimensions)):
            raise ValueError("对象标识和比较维度不得重复")
        if self.allowed_doc_ids and any(
            not set(o.doc_ids) <= set(self.allowed_doc_ids) for o in self.objects
        ):
            raise ValueError("对象关联文档超出允许范围")
        return self


class EvidenceFact(Model):
    fact_id: str = Field(default_factory=lambda: "f_" + uuid4().hex[:16])
    object_id: str
    dimension: str
    value: str = Field(min_length=1)
    unit: str = ""
    conditions: dict[str, str | None] = Field(default_factory=dict)
    sources: list[SourceRef] = Field(default_factory=list)
    status: Literal["candidate", "unsupported", "conflict", "superseded"] = "candidate"
    issue: str = ""


class Context(Model):
    chunk_id: str
    question: str
    summary: str
    score: int = Field(default=0, ge=0, le=10)
    fact_ids: list[str] = Field(default_factory=list)


class EvidenceBatch(Model):
    context: Context
    facts: list[EvidenceFact] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class SourceRead(Model):
    read_id: str
    chunk_ids: list[str]
    doc_id: str
    content: str
    pages: list[int]
    truncated: bool = False
    layout_uncertain: bool = False
    targets: list[str] = Field(default_factory=list)


class PQASession(Model):
    run_id: str = Field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8]
    )
    question: str
    spec: ComparisonSpec | None = None
    contexts: list[Context] = Field(default_factory=list)
    facts: list[EvidenceFact] = Field(default_factory=list)
    source_reads: list[SourceRead] = Field(default_factory=list)
    answer: str = ""
    explanation: str = ""
    answer_is_current: bool = False
    answer_data: dict[str, Any] = Field(default_factory=dict)
    stop_reason: str | None = None
    tool_history: list[list[str]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] = Field(
        default_factory=lambda: {
            "requests": 0,
            "failed_requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "missing_usage_requests": 0,
            "embedding_requests": 0,
            "embedding_tokens_unknown": 0,
        }
    )
    max_calls: int = 60
    max_tokens: int = 120000

    def add_tokens(self, prompt: int, completion: int) -> None:
        self.usage["prompt_tokens"] += prompt
        self.usage["completion_tokens"] += completion

    def record(self, event: str, **data: Any) -> None:
        self.events.append({"event": event, **data})

    def invalidate_answer(self) -> None:
        self.answer_is_current = False
        self.stop_reason = None
