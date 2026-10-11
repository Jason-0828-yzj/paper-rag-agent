# config.py —— 全项目统一的配置区
#
# 以后想调参数（比如换embedding模型、改chunk_size、换DeepSeek模型名），只改这一处，
# 其余模块（vectorstore.py / data_processing.py / rag_pipeline.py / agent_tools.py）全部引用这里，


import os
import sys
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 把 Hugging Face 的官方下载地址，强制替换成国内的镜像站地址

# PDF 里有 ↑、λ 这类特殊符号。万一控制台的编码装不下它们（比如 Windows 的 GBK 控制台），
# 就让它们显示成 "?"，而不是让程序报 UnicodeEncodeError 崩掉。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


# ====================== embedding / 切块配置 ======================
MODEL_NAME = "BAAI/bge-small-zh-v1.5"   # embedding 模型

# 切块参数。单位是"字符"，不是 token！
CHUNK_SIZE = 800        # 每块最多多少个字符
CHUNK_OVERLAP = 150     # 相邻两块之间重叠多少个字符
# Day8五轮实验（1000/200、500/50、800/150、1500/150、4000/200）挑出来的基线参数，
# 详见 Day8_评测记录.md 的"五轮实验总汇总"。

HERE = Path(__file__).resolve().parent                    # 本文件所在文件夹：...\py代码\rag实验

# 语料库默认放项目内 rag语料库/ 子文件夹（方便整体拷贝/clone）；
# 也可用环境变量 RAG_CORPUS_DIR 指向外部路径，优先级更高。
_default_corpus_dir = HERE / "rag语料库"
CORPUS_DIR = Path(os.environ.get("RAG_CORPUS_DIR", _default_corpus_dir))

# 索引目录名里带上切块参数：换参数重建时，新旧索引各存各的，不会互相覆盖，方便对比。
INDEX_DIR = HERE / f"faiss_index_c{CHUNK_SIZE}_o{CHUNK_OVERLAP}"


# ====================== 召回 / 精排配置（原 step6 / step7 共用的一份）======================
CANDIDATE_K = 20      # 召回阶段：先捞这么多候选
FINAL_K = 3           # 精排阶段：最终挑几个送进生成
RRF_K = 60            # 混合检索 RRF 融合公式里的平滑常数
SCORE_PREVIEW_CHARS = 400   # 打分阶段每块只给模型看多少字符


# ====================== DeepSeek API 配置 ======================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")   # 从环境变量里读取API密钥
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"
TEMPERATURE = 0.3     # 控制模型输出的随机性/创造力


if __name__ == "__main__":
    # 直接运行本文件 = 配置自检
    print("语料库目录：", CORPUS_DIR, "（存在）" if CORPUS_DIR.exists() else "（不存在！把PDF放进项目下的 rag语料库/ 文件夹，或设置环境变量 RAG_CORPUS_DIR 指向语料库位置）")
    print("索引目录：", INDEX_DIR, "（已存在）" if INDEX_DIR.exists() else "（还没建，先跑 data_processing.py）")
    print("DEEPSEEK_API_KEY：", "已设置" if DEEPSEEK_API_KEY else "未设置！")
