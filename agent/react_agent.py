from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .prompts import render_prompt
from .tool_executor import ToolExecutor
from .tools import format_observation_for_prompt


@dataclass
class ReActParseResult:
    thought: str = ""
    action_name: str = ""
    action_input: Dict[str, Any] | None = None
    final: str = ""
    raw: str = ""


_RE_THOUGHT = re.compile(r"(?mi)^\s*Thought\s*:\s*(.+)\s*$")
_RE_ACTION = re.compile(r"(?mi)^\s*Action\s*:\s*(.+)\s*$")
_RE_FINAL = re.compile(r"(?mi)^\s*Final\s*:\s*(.+)\s*$", re.DOTALL)


def _extract_first_json_value(text: str, *, start: int) -> str:
    """
    从 text[start:] 中提取第一个完整的 JSON 值（对象/数组），支持跨多行与 code fence。
    返回原始 JSON 字符串；失败则返回空字符串。
    """
    if not text:
        return ""
    start = max(0, int(start))

    brace = text.find("{", start)
    bracket = text.find("[", start)
    if brace < 0 and bracket < 0:
        return ""

    if brace < 0:
        begin = bracket
        open_ch, close_ch = "[", "]"
    elif bracket < 0:
        begin = brace
        open_ch, close_ch = "{", "}"
    else:
        begin = min(brace, bracket)
        if begin == brace:
            open_ch, close_ch = "{", "}"
        else:
            open_ch, close_ch = "[", "]"

    depth = 0
    in_str = False
    esc = False
    for i in range(begin, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == open_ch:
            depth += 1
            continue
        if ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[begin : i + 1].strip()
            continue

    return ""


def parse_react(text: str) -> ReActParseResult:
    out = ReActParseResult(raw=text or "")
    if not text:
        return out

    m_final = _RE_FINAL.search(text)
    if m_final:
        out.final = (m_final.group(1) or "").strip()
        return out

    m_thought = _RE_THOUGHT.search(text)
    if m_thought:
        out.thought = (m_thought.group(1) or "").strip()

    m_action = _RE_ACTION.search(text)
    if m_action:
        out.action_name = (m_action.group(1) or "").strip()

    # Action Input 允许多行 JSON（例如带缩进/换行），因此不能用单行正则强行截取。
    m_input = re.search(r"(?mi)^\s*Action Input\s*:\s*", text)
    if m_input:
        raw_json = _extract_first_json_value(text, start=m_input.end())
        try:
            parsed = json.loads(raw_json) if raw_json else None
            out.action_input = parsed if isinstance(parsed, dict) else None
        except Exception:
            out.action_input = None

    return out


class ReActAgent:
    def __init__(
        self,
        *,
        llm: Any,
        tool_executor: ToolExecutor,
        max_steps: int = 6,
        default_topk: int = 5,
        max_tool_calls: int = 2,
        llm_timeout_sec: int = 180,
    ) -> None:
        self.llm = llm
        self.tool_executor = tool_executor
        self.max_steps = max(1, int(max_steps))
        self.default_topk = max(1, int(default_topk))
        self.max_tool_calls = max(1, int(max_tool_calls))
        self.llm_timeout_sec = max(1, int(llm_timeout_sec))

    def _safe_print(self, text: str) -> None:
        try:
            print(text, flush=True)
        except UnicodeEncodeError:
            cleaned = re.sub(r"[\U00010000-\U0010ffff]", "", text)
            print(cleaned, flush=True)

    def _call_llm(self, prompt: str) -> str:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(self.llm.generate, prompt)
            try:
                out = fut.result(timeout=self.llm_timeout_sec)
            except FutureTimeoutError:
                return f"生成失败: LLM 调用超时（>{self.llm_timeout_sec}s）"
        return str(out or "")

    def run(self, question: str, *, show_steps: bool = False) -> Tuple[str, str]:
        history = ""
        tools_txt = self.tool_executor.format_tools_for_prompt()
        tool_call_count = 0

        for step in range(1, self.max_steps + 1):
            if show_steps:
                self._safe_print(f"\n--- 第 {step} 步 ---")
                self._safe_print("🧠 正在调用大语言模型...")

            prompt = render_prompt(tools=tools_txt, question=question, history=history)
            t0 = time.time()
            model_out = (self._call_llm(prompt) or "").strip()
            parsed = parse_react(model_out)

            if show_steps:
                self._safe_print(f"✅ 大语言模型响应（耗时 {time.time()-t0:.1f}s）:")
                if parsed.final:
                    if parsed.thought:
                        self._safe_print(f"Thought: {parsed.thought}")
                else:
                    self._safe_print(model_out)

            if parsed.final:
                if show_steps:
                    self._safe_print(f"🎉 最终答案: {parsed.final}")
                return parsed.final, history

            if not parsed.action_name:
                return model_out, history

            action_input = parsed.action_input or {}
            if "topk" not in action_input:
                action_input["topk"] = self.default_topk

            if tool_call_count >= self.max_tool_calls:
                obs = {"items": [], "error": "工具调用次数已达上限，请基于已有 Observation 输出 Final。"}
                obs_txt = "工具调用次数已达上限，请基于已有 Observation 输出 Final。"
            else:
                if show_steps:
                    self._safe_print(f"🎬 行动: {parsed.action_name}")
                    self._safe_print(f"🔍 正在执行工具，参数: {json.dumps(action_input, ensure_ascii=False)}")
                try:
                    obs = self.tool_executor.execute(name=parsed.action_name, arguments=action_input)
                except Exception as e:
                    obs = {"items": [], "error": str(e)}
                obs_txt = format_observation_for_prompt(obs) if isinstance(obs, dict) else str(obs)
                tool_call_count += 1
                if show_steps:
                    self._safe_print("👀 观察:")
                    self._safe_print(obs_txt)

            step_block = []
            if parsed.thought:
                step_block.append(f"Thought: {parsed.thought}")
            step_block.append(f"Action: {parsed.action_name}")
            step_block.append("Action Input: " + json.dumps(action_input, ensure_ascii=False))
            step_block.append("Observation:\n" + obs_txt)
            history = (history + "\n\n" + "\n".join(step_block)).strip()

        return "已达到最大步数，仍未得到 Final 输出。你可以尝试缩小问题范围或提高 topk。", history
