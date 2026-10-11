# vectorstore.py —— embedding 模型包装器 + 存/读 FAISS 向量库

from pathlib import Path
import os

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from config import MODEL_NAME, INDEX_DIR   # 配置统一从 config.py 来


class BgeEmbeddings(Embeddings):
    """把 SentenceTransformer 包成 LangChain 认识的 Embeddings。

    为什么要包一层：LangChain 要求"向量化器"必须有 embed_documents 和 embed_query 这两个方法。
    C++ 类比：Embeddings 是一个抽象基类（接口），这两个方法是纯虚函数；
              我们继承它（class BgeEmbeddings : public Embeddings），把两个方法实现掉。
    """

    def __init__(self, model_name: str = MODEL_NAME):      # 加载模型
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """入库时调用：一批文本 -> 一批向量（每块文本对应一个 512 维向量）。"""
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,   # 保证向量模长=1：模长=1 时"点积=余弦相似度"
            batch_size=32,               # 一次喂 32 块，GPU/CPU 都吃得消
            show_progress_bar=True,      # 显示进度条（控制台里的 Batches: xx%| 就是它，不是报错）
        )
        return vectors.tolist()          # numpy 数组 -> 普通 Python 列表（LangChain 要的格式）

    def embed_query(self, text: str) -> list[float]:
        """提问时调用：一个问题 -> 一个向量。"""
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()


def save_index(vectorstore, folder: Path = INDEX_DIR):
    """把 FAISS 库存到磁盘，生成 index.faiss（向量矩阵）和 index.pkl（原文+元数据+对应关系）两个文件。

    为什么不直接写 vectorstore.save_local(str(folder))：
    FAISS 底层是 C++ 写的，它自己打开文件时，在 Windows 上可能不认路径里的中文
    （FAISS 官方仓库 issue #3073 就是这个问题；路径里有"学习""代码""实验"）。
    绕法：先用 Python 的 os.chdir 走进目标文件夹，再让 FAISS 往当前目录 "." 里写，
    这样 C++ 那边看到的文件名只是纯英文的 index.faiss。
    """
    folder.mkdir(parents=True, exist_ok=True)      # 文件夹不存在就建一个
    old_cwd = os.getcwd()                          # 先记住"当前所在目录"
    os.chdir(folder)                               # 走进索引文件夹
    try:
        vectorstore.save_local(".")                # "." 表示当前目录；这行就是 LangChain 官方的存库写法
    finally:
        os.chdir(old_cwd)                          # 不管成功还是出错，最后都走回原来的目录


def load_index(embedder, folder: Path = INDEX_DIR):
    """把 data_processing.py 建好的 FAISS 库从磁盘读回来（读回来的是和建库时一模一样的库）。

    embedder 要传进去：库里只存了向量，没存"怎么把问题变成向量"，提问时还得靠它。
    建库和提问必须用同一个 embedding 模型，否则两边的向量不在同一个"空间"里，距离就没有意义。
    """
    if not (folder / "index.faiss").exists():
        raise SystemExit(
            f"还没有找到向量库：{folder}\n"
            "请先运行 data_processing.py 建库。"
            "（如果你改过 config.py 里的 CHUNK_SIZE / CHUNK_OVERLAP，需要用新参数重新建一次库。）"
        )
    old_cwd = os.getcwd()
    os.chdir(folder)                               # 同 save_index：绕开 FAISS 对中文路径的问题
    try:
        # allow_dangerous_deserialization=True：index.pkl 是 pickle 格式，pickle 在读取时可以执行任意代码，
        #   所以 LangChain 要求你明确声明"这个文件是我自己生成的、可信的"。
        #   这里没问题（文件是你自己用 data_processing.py 建的）；但以后读别人发给你的 .pkl，千万别随手写 True。
        return FAISS.load_local(".", embedder, allow_dangerous_deserialization=True)
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    # 直接运行本文件 = 路径自检
    from config import CORPUS_DIR
    print("语料库目录：", CORPUS_DIR, "（存在）" if CORPUS_DIR.exists() else "（不存在！检查文件夹位置）")
    print("其中 PDF 数量：", len(list(CORPUS_DIR.glob("*.pdf"))))
    print("索引将存到：", INDEX_DIR, "（已存在）" if INDEX_DIR.exists() else "（还没建，正常，先跑 data_processing.py）")
