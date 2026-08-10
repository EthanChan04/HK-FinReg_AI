"""
环境变量集中管理模块 (Configuration)
基于 Pydantic BaseSettings，自动从 .env 文件中加载配置。
包含 LangSmith 全链路追踪、CORS 安全策略与 API Key 认证配置。
"""
import json
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """应用全局配置"""
    # --- API Keys ---
    DEEPSEEK_API_KEY: str = ""
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
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"
    DEEPSEEK_TIMEOUT_SECONDS: int = 60
    DEEPSEEK_INTERACTIVE_THINKING: bool = False
    DEEPSEEK_REASONING_THINKING: bool = True
    EMBEDDING_PROVIDER: str = "local_hash"  # local_hash | openai_compatible
    EMBEDDING_MODEL: str = "local-hash"
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_DIMENSIONS: int = 256
    COPILOT_MAX_CONTEXT_CHARS: int = 16000
    COPILOT_MAX_HISTORY_MESSAGES: int = 8

    # --- RAG Config ---
    # Legacy fallback. Used only when SOURCE_MANIFEST_PATH is missing or empty.
    PDF_PATH: str = "../Fintech/AML Guideline for LCs_Eng_30 Sep 2021.pdf"
    CHROMA_COLLECTION: str = "hk_finreg_corpus"
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 200
    PARSER_MODE: str = "hierarchy"  # "hierarchy" | "reg_aware" | "flat" — M5 解析模式开关
    REG_DOC_DIR: str = "data/regulations"
    SOURCE_MANIFEST_PATH: str = "data/source_manifest.json"
    CORPUS_INDEX_DIR: str = "data/indexes"
    GRAPH_STORE_BACKEND: str = "networkx"
    GRAPH_STORE_PATH: str = "data/graph/regulatory_graph.json"
    RETRIEVAL_ROUTER_ENABLED: bool = True
    DEFAULT_RETRIEVAL_MODE: str = "rag"
    DEEP_RESEARCH_ENABLED: bool = True
    DEEP_RESEARCH_MAX_ITERATIONS: int = 3
    DEEP_RESEARCH_MAX_SUBQUESTIONS: int = 8
    DEEP_RESEARCH_MIN_EVIDENCE_PER_SUBQUESTION: int = 2
    SEMANTIC_CACHE_ENABLED: bool = False
    SEMANTIC_CACHE_THRESHOLD: float = 0.80
    SEMANTIC_CACHE_MAX_ENTRIES: int = 200
    SEMANTIC_CACHE_TTL_SECONDS: int = 3600
    SIRA_QUERY_PLANNER_ENABLED: bool = False
    SIRA_TERM_STATS_PATH: str = "data/indexes/term_statistics.json"
    EXPERIENCE_RAG_ENABLED: bool = False
    EXPERIENCE_RAG_MEMORY_PATH: str = "data/strategy_memory/retrieval_experiences.jsonl"
    EXPERIENCE_RAG_RECORDING_ENABLED: bool = False
    EXPERIENCE_RAG_MAX_RECORDS: int = 1000

    # --- Confidence Config (P2) ---
    CONFIDENCE_LOW_THRESHOLD: float = 0.5     # Rerank Top-1 < 0.5 → 低置信度警告
    CONFIDENCE_MED_THRESHOLD: float = 0.7     # Rerank Top-1 < 0.7 → 中等置信度提示
    CONFIDENCE_CROSS_VALIDATION_THRESHOLD: float = 0.3  # M3: retrieval vs reasoning 偏差阈值

    # --- App Config ---
    APP_TITLE: str = "HK-FinReg AI Backend"
    # CORS_ORIGINS: 支持从 .env 读取 JSON 数组字符串，如 '["https://your-domain.com"]'
    # 默认仅允许本地开发地址；生产部署必须通过 CORS_ORIGINS 环境变量配置实际域名
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    DEBUG: bool = False

    # --- Input Validation ---
    MAX_INPUT_LENGTH: int = 50000  # application_data 最大字符数

    # --- Security Config ---
    API_KEY_ENABLED: bool = True         # 设为 True 启用 API Key 认证（安全默认）
    API_KEY: str = ""                    # 在 .env 中设置: API_KEY=your_secret_key

    # --- Rate Limiting ---
    RATE_LIMIT_RPM: int = 60             # 每分钟最大请求数
    RATE_LIMIT_RPH: int = 500            # 每小时最大请求数

    # --- Workflow Checkpoint Config (Phase 1) ---
    WORKFLOW_CHECKPOINT_ENABLED: bool = True   # 是否启用工作流持久化
    WORKFLOW_DB_URL: str = ""                  # PostgreSQL 连接串，为空时 fallback 到 MemorySaver
    WORKFLOW_THREAD_PREFIX: str = "svf"        # workflow_run_id 前缀

    RATE_LIMIT_STORAGE_URL: str = ""
    TRUSTED_PROXY_HEADERS: bool = False

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # 允许 .env 存在冗余字段而不报错
    }

    @classmethod
    def _parse_cors_origins(cls, v):
        """支持从 .env 读取 JSON 数组字符串格式的 CORS_ORIGINS 环境变量"""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            # 逗号分隔 fallback
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


@lru_cache()
def get_settings() -> Settings:
    """获取全局唯一配置实例 (cached)"""
    return Settings()
