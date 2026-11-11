"""集中配置及 LMI 工厂；不在这里选择、合并证据。"""

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, PrivateAttr, model_validator

from paperqa.types import Model


class ParsingSettings(Model):
    chunk_chars: int = Field(default=1800, ge=200)
    overlap: int = Field(default=180, ge=0)
    table_context_chars: int = Field(default=2000, ge=0)
    table_layout_fallback: bool = False

    @model_validator(mode="after")
    def check_overlap(self):
        if self.overlap >= self.chunk_chars:
            raise ValueError("overlap 必须小于 chunk_chars")
        return self


class IndexSettings(Model):
    paper_directory: str = "papers"
    index_directory: str = ".paperqa-v1"
    index_name: str = "library"
    pending_uploads: int = Field(default=4, ge=1, le=16)
    document_hybrid: bool = False
    redis_url: str | None = None
    redis_ttl: int = Field(default=300, ge=1, le=3600)


class RerankSettings(Model):
    enabled: bool = False
    model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    revision: str = "1427fd652930e4ba29e8149678df786c240d8825"
    path: str | None = None
    local_files_only: bool = False
    candidates: int = Field(default=16, ge=2, le=48)
    evidence: int = Field(default=3, ge=1, le=8)
    timeout: float = Field(default=8, gt=0, le=45)


class AnswerSettings(Model):
    retrieval_mode: Literal["vector", "hybrid"] = "vector"
    evidence_k: int = Field(default=8, ge=1, le=40)
    max_sources: int = Field(default=12, ge=1)
    context_chars: int = Field(default=24000, ge=300)
    source_read_chars: int = Field(default=10000, ge=300)
    concurrency: int = Field(default=3, ge=1, le=12)
    mmr_lambda: float = Field(default=0.8, ge=0, le=1)
    rerank: RerankSettings = Field(default_factory=RerankSettings)


class AgentSettings(Model):
    model: str | None = None
    gap_repair: bool = False
    max_steps: int = Field(default=16, ge=1)
    max_calls: int = Field(default=60, ge=1)
    max_tokens: int = Field(default=120000, ge=100)
    timeout: float = Field(default=300, gt=0)
    search_count: int = Field(default=3, ge=1, le=20)
    concurrent_queries: int = Field(default=3, ge=1, le=12)


class Settings(Model):
    model: str = "deepseek/deepseek-flash"
    summary_model: str | None = None
    api_key_env: str = "DEEPSEEK_API_KEY"
    model_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "api_base": "https://api.deepseek.com",
            "temperature": 0,
            "extra_body": {"thinking": {"type": "disabled"}},
            "max_tokens": 3000,
            "max_retries": 0,
        }
    )
    embedding: str = "sparse"
    embedding_params: dict[str, Any] = Field(default_factory=dict)
    parsing: ParsingSettings = Field(default_factory=ParsingSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    answer: AnswerSettings = Field(default_factory=AnswerSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    _models: dict = PrivateAttr(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path | None = None):
        return (
            cls()
            if path is None
            else cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
        )

    def _get_model(self, name: str):
        from lmi import LiteLLMModel

        if name not in self._models:
            params = dict(self.model_params)
            if key := os.getenv(self.api_key_env):
                params["api_key"] = key
            self._models[name] = LiteLLMModel(
                name=name,
                config={
                    "model_list": [
                        {
                            "model_name": name,
                            "litellm_params": {
                                "model": name,
                                **params,
                            },
                        }
                    ],
                    # 每次请求由 core 计数；关闭 Router 隐藏重试。
                    "router_kwargs": {"num_retries": 0, "timeout": 90},
                },
            )
        return self._models[name]

    def get_llm(self):
        return self._get_model(self.model)

    def get_summary_llm(self):
        return self._get_model(self.summary_model or self.model)

    def get_agent_llm(self):
        return self._get_model(self.agent.model or self.model)

    def get_embedding_model(self):
        from paperqa.llms import embedding_model_factory

        return embedding_model_factory(self.embedding, self.embedding_params)
