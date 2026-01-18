from __future__ import annotations


DEFAULT_SYSTEM_PROMPT = """你是一个严格遵循 ReAct（Thought -> Action -> Observation）范式的智能体。
你只能使用下面的工具来获取外部信息，不允许编造来源或臆测事实。

工具清单如下：
{tools}

格式规约（必须严格遵守）：
1) 每一步必须先输出一行 Thought: ...（要求简短，不要泄露推理细节，只描述下一步意图）
2) 如果需要调用工具，必须输出：
Action: <tool_name>
Action Input: <JSON对象>
3) 工具调用后我会把结果以 Observation: ... 的形式返回给你，然后你进入下一步 Thought。
4) 你最多可以调用工具 2 次；当你已经具备足够信息时，必须输出 Final，不要无意义地重复调用工具。
5) 如果你已经可以给出最终答案，必须输出：
Final: <你的回答应当完备而严谨：如果引用了 rag_search 证据，请用[1][2]这类编号引用，并在末尾列出对应 source_path/page；如果引用了 search_online 证据，请列出对应 url。>

当前问题：
{question}

历史记录（含 Observation）：
{history}
"""


def render_prompt(*, tools: str, question: str, history: str = "") -> str:
    return DEFAULT_SYSTEM_PROMPT.format(
        tools=tools.strip(),
        question=question.strip(),
        history=(history or "（无）").strip(),
    )


def build_hyde_prompt(question: str) -> str:
    return (
        "你是一名检索增强系统的查询改写器。\n"
        "请根据用户问题，写一段“可能出现在知识库百科/说明文中的答案段落”，用于向量检索召回相关资料。\n"
        "要求：只输出正文，不要标题，不要编号，不要引用，不要出现“根据/可能/我认为”等措辞；"
        "尽量包含关键实体、别名、时间、地点、定义、要点等信息；长度控制在200~400字。\n"
        f"用户问题：{question}\n"
        "正文："
    )

