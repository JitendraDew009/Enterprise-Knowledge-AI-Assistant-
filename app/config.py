from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_knowledge"
    environment: str = "development"
    log_level: str = "INFO"
    max_upload_bytes: int = Field(default=10_000_000, gt=0)
    retrieval_score_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    llm_timeout_seconds: int = Field(default=30, gt=0)
    llm_max_retries: int = Field(default=1, ge=0)
    conversation_history_limit: int = Field(default=10, ge=1, le=50)
    api_base_url: str = "http://127.0.0.1:8000"
    application_api_key: str | None = None
    cors_origins: str = "http://127.0.0.1:8000,http://localhost:8501"
    vector_backend: Literal["pgvector", "chroma"] = "pgvector"
    jwt_secret: str | None = None
    jwt_algorithm: str = "HS256"
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    chroma_persist_directory: str = "./data/chroma"
    embedding_dimensions: int = Field(default=1536, gt=0)
    retrieval_k: int = Field(default=4, ge=1, le=20)
    chunk_size: int = Field(default=900, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {', '.join(sorted(allowed))}.")
        return normalized

    @field_validator("vector_backend", mode="before")
    @classmethod
    def normalize_vector_backend(cls, value: str) -> str:
        return value.lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_foundation_settings(self) -> "Settings":
        if (
            "chunk_size" in self.model_fields_set
            and "chunk_overlap" in self.model_fields_set
            and self.chunk_overlap >= self.chunk_size
        ):
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if self.vector_backend == "pgvector" and self.embedding_dimensions != 1536:
            raise ValueError("EMBEDDING_DIMENSIONS must be 1536 when VECTOR_BACKEND=pgvector.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
