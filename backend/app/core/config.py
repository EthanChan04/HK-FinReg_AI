"""
鐜鍙橀噺闆嗕腑绠＄悊妯″潡 (Configuration)
鍩轰簬 Pydantic BaseSettings锛岃嚜鍔ㄤ粠 .env 鏂囦欢涓姞杞介厤缃€?
鍖呭惈 LangSmith 鍏ㄩ摼璺拷韪€丆ORS 瀹夊叏绛栫暐涓?API Key 璁よ瘉閰嶇疆銆?
"""
import json
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """搴旂敤鍏ㄥ眬閰嶇疆"""
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
    LANGSMITH_TRACING: bool = True  # 鎬诲紑鍏?
    LANGCHAIN_TRACING_V2: str | None = None
    LANGCHAIN_ENDPOINT: str | None = None
    LANGCHAIN_API_KEY: str | None = None
    LANGCHAIN_PROJECT: str | None = None

    # --- Model Config ---
    ZHIPU_MODEL: str = "MiMo-v2.5"
    ZHIPU_BASE_URL: str = "https://token-plan-cn.xiaomimimo.com/v1"
    ZHIPU_EMBEDDING_MODEL: str = "embedding-3"
    EMBEDDING_PROVIDER: str = "openai_compatible"  # openai_compatible | local_hash
    EMBEDDING_MODEL: str = "embedding-3"
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_DIMENSIONS: int = 256
    LONGCAT_MODEL: str = "MiMo-v2.5"
    LONGCAT_BASE_URL: str = "https://token-plan-cn.xiaomimimo.com/v1"
    LLM_TIMEOUT_SECONDS: int = 60  # LLM API timeout in seconds
    COPILOT_MODEL: str = "MiMo-v2.5"
    COPILOT_BASE_URL: str = "https://token-plan-cn.xiaomimimo.com/v1"
    COPILOT_API_KEY: str = ""
    COPILOT_TIMEOUT_SECONDS: int = 60
    COPILOT_MAX_CONTEXT_CHARS: int = 16000
    COPILOT_MAX_HISTORY_MESSAGES: int = 8

    # --- RAG Config ---
    # Legacy fallback. Used only when SOURCE_MANIFEST_PATH is missing or empty.
    PDF_PATH: str = "../Fintech/AML Guideline for LCs_Eng_30 Sep 2021.pdf"
    CHROMA_COLLECTION: str = "hk_finreg_corpus"
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 200
    PARSER_MODE: str = "hierarchy"  # "hierarchy" | "reg_aware" | "flat" 鈥?M5 瑙ｆ瀽妯″紡寮€鍏?
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

    # --- Confidence Config (P2) ---
    CONFIDENCE_LOW_THRESHOLD: float = 0.5     # Rerank Top-1 < 0.5 鈫?浣庣疆淇″害璀﹀憡
    CONFIDENCE_MED_THRESHOLD: float = 0.7     # Rerank Top-1 < 0.7 鈫?涓瓑缃俊搴︽彁绀?
    CONFIDENCE_CROSS_VALIDATION_THRESHOLD: float = 0.3  # M3: retrieval vs reasoning 鍋忓樊闃堝€?

    # --- App Config ---
    APP_TITLE: str = "HK-FinReg AI Backend"
    # CORS_ORIGINS: 鏀寔浠?.env 璇诲彇 JSON 鏁扮粍瀛楃涓诧紝濡?'["https://your-domain.com"]'
    # 榛樿浠呭厑璁告湰鍦板紑鍙戝湴鍧€锛涚敓浜ч儴缃插繀椤婚€氳繃 CORS_ORIGINS 鐜鍙橀噺閰嶇疆瀹為檯鍩熷悕
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    DEBUG: bool = False

    # --- Input Validation ---
    MAX_INPUT_LENGTH: int = 50000  # application_data 鏈€澶у瓧绗︽暟

    # --- Security Config ---
    API_KEY_ENABLED: bool = True         # 璁句负 True 鍚敤 API Key 璁よ瘉锛堝畨鍏ㄩ粯璁わ級
    API_KEY: str = ""                    # 鍦?.env 涓缃? API_KEY=your_secret_key

    # --- Workflow Checkpoint Config (Phase 1) ---
    WORKFLOW_CHECKPOINT_ENABLED: bool = True   # 鏄惁鍚敤宸ヤ綔娴佹寔涔呭寲
    WORKFLOW_DB_URL: str = ""                  # PostgreSQL 杩炴帴涓诧紝涓虹┖鏃?fallback 鍒?MemorySaver
    WORKFLOW_THREAD_PREFIX: str = "svf"        # workflow_run_id 鍓嶇紑

    model_config = {
        "env_file": ".env", 
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # 鍏佽 .env 瀛樺湪鍐椾綑瀛楁鑰屼笉鎶ラ敊
    }

    @classmethod
    def _parse_cors_origins(cls, v):
        """鏀寔 JSON 鏁扮粍瀛楃涓叉牸寮忕殑 CORS_ORIGINS 鐜鍙橀噺"""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            # 閫楀彿鍒嗛殧 fallback
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


@lru_cache()
def get_settings() -> Settings:
    """鑾峰彇鍏ㄥ眬鍞竴閰嶇疆瀹炰緥 (cached)"""
    return Settings()


