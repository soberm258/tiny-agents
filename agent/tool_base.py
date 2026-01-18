from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str


class BaseTool(ABC):
    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        raise NotImplementedError

    def prompt_usage(self) -> str:
        """
        返回给 LLM 的工具使用说明（可包含参数格式）。
        子类可覆盖以提供更清晰的 schema。
        """
        return ""

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def normalize_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if arguments is None:
            return {}
        if not isinstance(arguments, dict):
            raise TypeError("arguments 必须是 dict")
        return arguments

