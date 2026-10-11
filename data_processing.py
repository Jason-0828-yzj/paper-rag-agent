# 论文 PDF -> 切块 -> 向量化 -> 存进 FAISS
# （RAG 的"离线建库"阶段：只需要做一次，建好以后每次提问都直接用这个库）
#
# 流水线：
#   [1] PyPDFLoader 读 PDF                      -> 每一页变成一个 Document（文本 + 元数据）
#   [2] RecursiveCharacterTextSplitter 切块     -> 每页再切成若干个小 Document（chunk）
#   [3] embedding 模型把每块文本变成向量           -> BgeEmbeddings
#   [4] 向量 + 原文一起存进 FAISS，再存到磁盘       -> save_index（里面是 LangChain 的 save_local）
#
# 想调切块参数：去 config.py 的配置区改 CHUNK_SIZE / CHUNK_OVERLAP，再重新运行本文件。

import time
from collections import Counter
from pathlib import Path

# 先 import config：它里面设置了 HF 镜像，要赶在别的库之前
from config import CORPUS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, INDEX_DIR
from vectorstore import BgeEmbeddings, save_index

from langchain_community.document_loaders import PyPDFLoader               # PDF 加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter        # 切块器
from langchain_community.vectorstores import FAISS                         # 向量库（LangChain 对 FAISS 的封装）

# ---------------------------------------------------------------- [0] 加载 embedding 模型
print("[0/4] 加载 embedding 模型 ...")
embedder = BgeEmbeddings()
print(f"  模型跑在：{embedder.model.device}（cuda:0 = 用显卡，cpu = 用 CPU）")

# ---------------------------------------------------------------- [1] 加载 PDF
print("\n[1/4] 加载 PDF ...")
pdf_files = sorted(CORPUS_DIR.glob("*.pdf"))      # 只要 .pdf
if not pdf_files:
    raise SystemExit(f"没找到 PDF，请检查路径：{CORPUS_DIR}")

docs = []
for pdf in pdf_files:
    pages = PyPDFLoader(str(pdf)).load()          # 返回 list[Document]：PDF 有几页，列表里就有几个元素
    n_chars = sum(len(p.page_content) for p in pages)
    print(f"  {pdf.name}: {len(pages)} 页, {n_chars} 字符")
    docs.extend(pages)
print(f"  合计：{len(pdf_files)} 篇，{len(docs)} 页")

# ---------------------------------------------------------------- [2] 切块
print(f"\n[2/4] 切块：chunk_size={CHUNK_SIZE} 字符, chunk_overlap={CHUNK_OVERLAP} 字符 ...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True, # 记录这一块在原文的哪个位置，方便定位和调试
)
chunks = splitter.split_documents(docs)

char_lens = [len(c.page_content) for c in chunks]
print(f"  共 {len(chunks)} 块；字符数 平均 {sum(char_lens) / len(char_lens):.0f} / 最短 {min(char_lens)} / 最长 {max(char_lens)}")

limit = embedder.model.max_seq_length # 512
tokenizer = embedder.model.tokenizer
token_lens = [len(ids) for ids in tokenizer([c.page_content for c in chunks],
                                            truncation=False, verbose=False)["input_ids"]]
too_long = sum(1 for n in token_lens if n > limit) # 统计有多少块的token数超过了模型上限
print(f"  token 数 平均 {sum(token_lens) / len(token_lens):.0f} / 最长 {max(token_lens)}；"
      f"超过模型上限 {limit} 的块：{too_long} 块（占 {too_long / len(chunks):.1%}）")
if too_long:
    print("  [提示] 这些块超出了 bge 能读的长度：超出的尾部不会体现在向量里，检索时等于'看不见'。")
    longest = max(range(len(chunks)), key=lambda i: token_lens[i])
    c = chunks[longest]
    print(f"  最长的一块：{token_lens[longest]} 个 token（只有 {len(c.page_content)} 个字符），来自 "
          f"{Path(c.metadata['source']).name} 第 {c.metadata['page'] + 1} 页，开头是：")
    print("   ", repr(c.page_content[:200]))

per_paper = Counter(Path(c.metadata["source"]).name for c in chunks)
for name, n in sorted(per_paper.items()):
    print(f"    {name}: {n} 块")

# ---------------------------------------------------------------- [3] 向量化 + 存入 FAISS
print("\n[3/4] 向量化并建立 FAISS 索引（这一步最花时间）...")
t0 = time.time()
vectorstore = FAISS.from_documents(chunks, embedder)
print(f"  完成，用时 {time.time() - t0:.1f} 秒；库里共 {vectorstore.index.ntotal} 个向量，每个 {vectorstore.index.d} 维")

# ---------------------------------------------------------------- [4] 存到磁盘
print("\n[4/4] 保存到磁盘 ...")
save_index(vectorstore)
for f in sorted(INDEX_DIR.iterdir()):
    print(f"  {f.name}: {f.stat().st_size / 1024:.0f} KB") # 字节 ->KB
print(f"  已保存到：{INDEX_DIR}")

print("\n建库完成。")
