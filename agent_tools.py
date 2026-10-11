# agent_tools.py —— 给RAG项目加Agent工具调用（ReAct范式）
#
# 把"召回+精排"这部分逻辑抽掉了（现在从 rag_pipeline.py 导入 recall/rerank，不再在本文件里重复定义），
# 只保留这个文件真正该负责的东西：两个 @tool 工具的定义 + create_agent() 的组装。
#
# 背景（原样保留，说明为什么要做这件事）：
#   step1~step6把"检索"这件事从向量检索做到了混合检索+LLM精排，但这条链路从头到尾
#   都是"你（人）决定要不要检索、检索几次、用什么问法"。Day9要做的事情，是把"要不要用
#   工具、用什么参数调用"这个决策权交给大模型自己——这就是 Function Calling（工具调用）：
#   1) 你把一个或多个"工具"（本质就是普通Python函数，附带一段说明文字）告诉大模型；
#   2) 大模型看到用户问题后，自己判断"这个问题需不需要调用某个工具、该传什么参数"；
#   3) 如果需要，大模型不会直接给出最终答案，而是输出一个"工具调用请求"（函数名+参数）；
#   4) 你的代码把这个请求实际执行一遍，把结果（Observation）喂回给大模型；
#   5) 大模型拿到结果后继续"推理"，可能再调用一次工具，也可能这时候才给出最终答案。
#   这个"推理(Reasoning) -> 行动(Acting) -> 观察结果 -> 再推理 -> ……"反复交替、直到给出
#   最终答案的过程，就是 ReAct 范式。
#
# 两个工具：
#   1) retrieve_from_papers —— "二次检索"工具：把 rag_pipeline 里验证过的混合检索+LLM精排
#      链路包成一个工具，让大模型自己决定要不要拆解问题、调用几次。
#   2) calculator —— 计算器工具：涉及数值比较/换算时调用，避免大模型口算出错。

from pathlib import Path

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, TEMPERATURE, CANDIDATE_K, FINAL_K
import rag_pipeline

# ---- DeepSeek接入LangChain：OpenAI兼容格式API，直接用 ChatOpenAI 类换个 base_url 即可 ----
if not DEEPSEEK_API_KEY:
    raise SystemExit("没读到 DEEPSEEK_API_KEY 这个环境变量，请为这个脚本设置一下。")

llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    base_url=DEEPSEEK_BASE_URL,
    api_key=DEEPSEEK_API_KEY,
    temperature=TEMPERATURE,
)


# =====================================================================================
#                                 工具定义
# =====================================================================================

@tool
def retrieve_from_papers(query: str) -> str:
    """从本地论文库中检索与问题最相关的原文片段。

    何时使用：
    - 当用户的问题需要查阅具体论文内容（如方法细节、实验数据、结论）时，必须调用此工具。
    - 当用户只是闲聊、做数学计算、或问常识性问题时，不要调用此工具。

    参数 query 的写法：
    - 请传入一个完整、具体的问题或子问题，不要只传几个关键词。
    - 如果原问题涉及多篇论文的比较，建议拆成针对单篇论文的子问题，分多次调用。
    """
    candidates = rag_pipeline.recall(query, k=CANDIDATE_K)        # 召回阶段
    reranked, scores = rag_pipeline.rerank(query, candidates, llm)  # 精排阶段
    top_docs = reranked[:FINAL_K]
    top_scores = scores[:FINAL_K]

    parts = []
    for doc, score in zip(top_docs, top_scores):
        name = Path(doc.metadata["source"]).name
        page = doc.metadata["page"] + 1
        parts.append(f"[来源：{name} 第{page}页，相关性{score:.1f}/10分]\n{doc.page_content}")
    return "\n\n".join(parts)


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (supports + - * / and parentheses,
    e.g. "34.5 / 0.6" or "(79.0 - 75.3) / 75.3 * 100") and return the numeric result.
    Use this whenever the question requires comparing, dividing, or computing a
    percentage/ratio between numbers you already know (e.g. from retrieve_from_papers)."""
    import re
    # 先校验 expression 只含安全字符，再 eval
    if not re.fullmatch(r"[0-9.+\-*/() ]+", expression):
        return f"这不是一个合法的算术表达式：{expression!r}，只能包含数字、小数点、+-*/()和空格。"
    try:
        result = eval(expression)
    except Exception as e:
        return f"计算出错：{type(e).__name__}: {e}"
    return str(result)


def check_calculator():
    """自检：calculator 写得对不对。"""
    ok = calculator.invoke({"expression": "34.5 / 0.6"})
    if "57.5" not in ok:
        raise SystemExit(f"34.5/0.6 应该约等于57.5，但算出了：{ok}")

    blocked = calculator.invoke({"expression": "__import__('os').system('echo hi')"})
    if "合法" not in blocked and "非法" not in blocked and "不是" not in blocked:
        raise SystemExit(
            f"危险输入 __import__(...) 没有被拦截，直接eval了！返回了：{blocked}"
        )


# =====================================================================================
#                                   组装 Agent
# =====================================================================================
agent = create_agent(
    model=llm,
    tools=[retrieve_from_papers, calculator],
    system_prompt=(
        "你是一个论文问答助手，负责回答关于几篇AI论文（D-HGNet/SlimDy-YOLO、DySample、"
        "consistent dual assignments、ParameterNet、Slim-Neck/GSConv、VoVNet、SSLD、"
        "CondConv等）的问题。你有两个工具可以调用：\n"
        "1) retrieve_from_papers：查询论文原文内容。如果问题涉及多篇论文的比较，"
        "先把问题拆成针对单篇论文的子问题，分别调用这个工具，再综合各次检索结果作答。\n"
        "2) calculator：计算数值表达式。涉及比较/换算/百分比计算时使用，不要自己口算数字。\n"
        "必须严格根据工具返回的内容作答，不能使用你自己固有的知识编造论文细节。"
        "如果检索结果不包含答案，直接说明'资料中没有相关信息'，不要猜测。"
    ),
)


def run_agent(question: str):
    """跑一次Agent，把每一步的"推理/行动/观察"轨迹都打印出来，方便看ReAct是怎么交替的。"""
    print("\n" + "=" * 90)
    print("问题：", question)

    inputs = {"messages": [{"role": "user", "content": question}]}
    for chunk in agent.stream(inputs, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            messages = node_output.get("messages", [])
            for msg in messages:
                role = getattr(msg, "type", msg.__class__.__name__)
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for call in tool_calls:
                        print(f"  [行动 Action] 调用工具 {call['name']}，参数：{call['args']}")
                elif role == "tool":
                    name = getattr(msg, "name", "?")
                    content_preview = str(msg.content)[:200].replace("\n", " ")
                    print(f"  [观察 Observation] 工具 {name} 返回：{content_preview} ...")
                else:
                    content = getattr(msg, "content", "")
                    if content:
                        print(f"  [{role}] {content}")


if __name__ == "__main__":
    check_calculator()
    print("calculator 自检通过（正常算术能算，危险输入被拦截）。")
