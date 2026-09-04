from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_knowledge"
    environment: str = "development"
    log_level: str = "INFO"
    max_upload_bytes: int = 10_000_000
    retrieval_score_threshold: float = 0.25
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 1
    conversation_history_limit: int = 10
    api_base_url: str = "http://127.0.0.1:8000"
    application_api_key: str | None = None
    cors_origins: str = "http://127.0.0.1:8000,http://localhost:8501"
    vector_backend: str = "chroma"
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    chroma_persist_directory: str = "./data/chroma"
    retrieval_k: int = 4
    chunk_size: int = 900
    chunk_overlap: int = 150

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
