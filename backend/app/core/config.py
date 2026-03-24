"""
环境变量集中管理模块 (Configuration)
基于 Pydantic BaseSettings，自动从 .env 文件中加载配置。
包含 LangSmith 全链路追踪配置。
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """应用全局配置"""
    # --- API Keys ---
    ZHIPU_API_KEY: str = ""
    LONGCAT_API_KEY: str = ""
    DASHSCOPE_API_KEY: str | None = None
    COHERE_API_KEY: str = ""

    # --- Reranker Config ---
    COHERE_RERANK_MODEL: str = "rerank-v3.5"
    RERANK_TOP_K: int = 5

    # --- LangSmith Tracing ---
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "hk-finreg-ai"
    LANGSMITH_TRACING: bool = True  # 总开关
    LANGCHAIN_TRACING_V2: str | None = None
    LANGCHAIN_ENDPOINT: str | None = None
    LANGCHAIN_API_KEY: str | None = None
    LANGCHAIN_PROJECT: str | None = None

    # --- Model Config ---
    ZHIPU_MODEL: str = "glm-4-flash"
    ZHIPU_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4/"
    ZHIPU_EMBEDDING_MODEL: str = "embedding-3"
    LONGCAT_MODEL: str = "LongCat-Flash-Thinking-2601"
    LONGCAT_BASE_URL: str = "https://api.longcat.chat/openai/v1"

    # --- RAG Config ---
    PDF_PATH: str = "../Fintech/AML Guideline for LCs_Eng_30 Sep 2021.pdf"
    CHROMA_COLLECTION: str = "zhipu_collection"
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 200

    # --- App Config ---
    APP_TITLE: str = "HK-FinReg AI Backend"
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000", "*"]
    DEBUG: bool = False

    model_config = {
        "env_file": ".env", 
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # 允许 .env 存在冗余字段而不报错
    }


@lru_cache()
def get_settings() -> Settings:
    """获取全局唯一配置实例 (cached)"""
    return Settings()
