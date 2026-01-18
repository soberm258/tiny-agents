import sys

sys.path.append(".")

from agent.react_agent import parse_react


def test_parse_react_multiline_action_input() -> None:
    text = """Thought: 需要检索相关证据
Action: rag_search
Action Input:
{
  "query": "南京是什么",
  "topk": 3
}
"""
    r = parse_react(text)
    assert r.final == ""
    assert r.action_name == "rag_search"
    assert r.action_input == {"query": "南京是什么", "topk": 3}


def test_parse_react_action_input_code_fence() -> None:
    text = """Thought: 先检索
Action: rag_search
Action Input: ```json
{
  "query": "北京",
  "topk": 5
}
```
"""
    r = parse_react(text)
    assert r.action_name == "rag_search"
    assert r.action_input == {"query": "北京", "topk": 5}


def test_parse_react_final_first() -> None:
    text = """Thought: 已足够
Final: 这是最终答案
Action: rag_search
Action Input: {"query": "不会被解析", "topk": 1}
"""
    r = parse_react(text)
    assert r.final.startswith("这是最终答案")
    assert r.action_name == ""
    assert r.action_input is None


if __name__ == "__main__":
    test_parse_react_multiline_action_input()
    test_parse_react_action_input_code_fence()
    test_parse_react_final_first()
    print("ok")
