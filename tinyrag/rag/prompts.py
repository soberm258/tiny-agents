from __future__ import annotations


RAG_PROMPT_TEMPLATE = """参考信息（每段以 [编号] 开头）：
{context}
---
我的问题或指令：
{question}
---
我的回答：
{answer}
---
请根据上述参考信息回答和我的问题或指令，修正我的回答。前面的参考信息和我的回答可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你修正的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。请在关键结论后标注引用编号，例如 [1][3]。
你修正的回答："""


HYDE_PROMPT_TEMPLATE = """你是一名检索增强系统的查询改写器。
请根据用户问题，写一段“可能出现在知识库/百科/说明文中的答案段落”，用于向量检索召回相关资料。
要求：只输出正文，不要标题，不要编号，不要引用，不要出现“根据/可能/我认为”等措辞；尽量包含关键实体、别名、时间、地点、定义、要点等信息；长度控制在 200~400 字。
用户问题：{question}
正文："""


def build_rag_prompt(*, context: str, question: str, answer: str) -> str:
    return RAG_PROMPT_TEMPLATE.format(
        context=(context or "").strip(),
        question=(question or "").strip(),
        answer=(answer or "").strip(),
    )


def build_hyde_prompt(question: str) -> str:
    return HYDE_PROMPT_TEMPLATE.format(question=(question or "").strip())

