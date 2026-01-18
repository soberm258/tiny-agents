from __future__ import annotations

from typing import Any, Dict, List

from .tool_base import BaseTool, ToolSpec


class ToolExecutor:
    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        name = (tool.spec.name or "").strip()
        if not name:
            raise ValueError("工具 name 不能为空")
        if name in self._tools:
            raise ValueError(f"工具已注册：{name}")
        self._tools[name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"未注册工具：{name}")
        return self._tools[name]

    def list_specs(self) -> List[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def format_tools_for_prompt(self) -> str:
        blocks: List[str] = []
        for tool in self._tools.values():
            usage = tool.prompt_usage().strip()
            if usage:
                blocks.append(f"Name: {tool.spec.name}\nDescription: {tool.spec.description}\nUsage:\n{usage}")
            else:
                blocks.append(f"Name: {tool.spec.name}\nDescription: {tool.spec.description}")
        return "\n\n".join(blocks)

    def execute(self, *, name: str, arguments: Dict[str, Any]) -> Any:
        tool = self.get(name)
        arguments = tool.normalize_arguments(arguments)
        return tool.run(**arguments)

