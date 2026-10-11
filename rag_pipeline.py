# rag_pipeline.py —— RAG检索核心链路：混合检索(召回) + LLM打分(精排)
#
# "被Agent当工具调用"的链路都从这里导入。
#
# 两阶段设计（详见 Day8_评测记录.md"重排序"实验）：
#   第一阶段【召回/recall】：向量检索 + BM25关键词检索，用RRF融合排名，先捞一批"大概率相关"
#     的候选（CANDIDATE_K个）——这一步要的是"宁可错杀不可放过"。
#   第二阶段【精排/rerank】：用大模型给每个候选块打相关性分数，重新排序，挑出最终 FINAL_K 个。
#     这里用"LLM当cross-encoder"的思路代替专门训练的 reranker 模型。

import re

import numpy as np
from rank_bm25 import BM25Okapi

from config import CANDIDATE_K, FINAL_K, RRF_K, SCORE_PREVIEW_CHARS
from vectorstore import BgeEmbeddings, load_index


# ---- 加载向量库 + 建 BM25 索引（模块级初始化：import 这个文件时只跑一次）----
embedder = BgeEmbeddings()
vs = load_index(embedder)
N = vs.index.ntotal
print(f"库已加载：共 {N} 块。")


def get_all_chunks(vectorstore):
    chunks = []
    for i in range(N):
        doc_id = vectorstore.index_to_docstore_id[i]
        doc = vectorstore.docstore.search(doc_id)
        chunks.append(doc)
    return chunks


all_chunks = get_all_chunks(vs)

# 分词：连续英文/数字当一个词，中文一个字一个词；标点/空格/横杠全扔掉
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+|[一-鿿]", text.lower())


tokenized_corpus = [tokenize(doc.page_content) for doc in all_chunks]
bm25 = BM25Okapi(tokenized_corpus)   # 提前算好"每个词的权重"，建好索引


def scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    """得分 -> 排名"""
    order = np.argsort(-scores)
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def recall(question: str, k: int = CANDIDATE_K):
    """召回阶段：向量检索 + BM25检索，RRF融合排名，返回前k个候选块。"""
    vectors = np.array(vs.index.reconstruct_n(0, N))
    q = np.array(embedder.embed_query(question))
    vector_scores = vectors @ q
    bm25_scores = np.array(bm25.get_scores(tokenize(question)))

    vector_ranks = scores_to_ranks(vector_scores)
    bm25_ranks = scores_to_ranks(bm25_scores)
    fused_scores = 1.0 / (RRF_K + vector_ranks) + 1.0 / (RRF_K + bm25_ranks)

    top_rows = np.argsort(-fused_scores)[:k]
    return [all_chunks[row] for row in top_rows]


def build_rerank_prompt(question: str, candidates: list):
    """把召回阶段的N个候选块编号列出来，让模型给每一块打0~10分的相关性分数。"""
    lines = []
    for i, doc in enumerate(candidates, start=1):
        preview = doc.page_content[:SCORE_PREVIEW_CHARS].replace("\n", " ")
        lines.append(f"[{i}] {preview}")
    candidates_text = "\n".join(lines)

    system_prompt = (
        "你是一个检索结果打分助手。给定一个问题和若干候选文本块，请判断每个候选块对回答这个"
        "问题的帮助程度，打0到10分（0分=完全无关，10分=直接包含完整答案）。"
        "严格按照下面的格式输出，每行一个，不要有任何多余的文字、解释或标题：\n"
        "编号:分数\n"
        "例如：\n1:8\n2:0\n3:5\n"
        "必须给每一个候选块都打分，不能遗漏，也不能编造不存在的编号。"
    )
    user_prompt = f"问题：{question}\n\n候选块：\n{candidates_text}"
    return system_prompt, user_prompt


def parse_scores(reply: str, n: int) -> list[float]:
    """用正则从模型回复里提取"编号:分数"对，按编号1~n的顺序整理成列表返回。"""
    pairs = re.findall(r"(\d+)\s*[：:]\s*(\d+(?:\.\d+)?)", reply)
    score_map = {}
    for idx_str, score_str in pairs:
        score_map[int(idx_str)] = float(score_str)
    return [score_map[i] for i in range(1, n + 1)]


def rerank(question: str, candidates: list, llm):
    """精排阶段：调用llm给所有候选打分，按分数重新排序。

    llm 由调用者传入（而不是在本模块里创建）——因为 step6 原本用的是 requests.post
    手写调用、step7 用的是 LangChain 的 ChatOpenAI，两种调用方式不一样，这里统一成
    "调用方传一个具备 .invoke([...]).content 接口的对象进来"，哪种风格都能用。
    """
    # system_prompt：给大模型立规矩（按 编号:分数 格式打分）。
    # user_prompt：把问题和候选块列出来。
    system_prompt, user_prompt = build_rerank_prompt(question, candidates)
    reply = llm.invoke(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    ).content
    scores = parse_scores(reply, len(candidates))
    order = np.argsort(-np.array(scores))
    return [candidates[i] for i in order], [scores[i] for i in order] # （排序后的候选块列表，排序后的分数列表）
