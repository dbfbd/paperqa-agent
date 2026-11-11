"""小型公开入口；解析插件导入时不会提前装配 Agent。"""

from importlib import import_module

__all__ = ["ComparisonSpec", "Docs", "PQASession", "Settings", "agent_query"]
_EXPORTS = {
    "Docs": ("paperqa.docs", "Docs"),
    "Settings": ("paperqa.settings", "Settings"),
    "agent_query": ("paperqa.agents.main", "agent_query"),
    "ComparisonSpec": ("paperqa.types", "ComparisonSpec"),
    "PQASession": ("paperqa.types", "PQASession"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attr = _EXPORTS[name]
    return getattr(import_module(module), attr)
