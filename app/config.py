"""Application configuration."""

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a .env file."""

    app_name: str = Field(
        default="Agentic API Troubleshooting Assistant",
        alias="APP_NAME",
    )
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME",
    )
    embedding_batch_size: int = Field(default=32, ge=1, alias="EMBEDDING_BATCH_SIZE")
    chroma_persist_dir: Path = Field(default=Path("chroma_db"), alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(
        default="order-service-knowledge",
        alias="CHROMA_COLLECTION_NAME",
    )
    vector_upsert_batch_size: int = Field(default=64, ge=1, alias="VECTOR_UPSERT_BATCH_SIZE")
    retrieval_top_k: int = Field(default=5, ge=1, alias="RETRIEVAL_TOP_K")

    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-5-nano", alias="OPENAI_MODEL")
    llm_timeout_seconds: float = Field(default=60.0, gt=0, alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=2, ge=0, alias="LLM_MAX_RETRIES")

    agent_max_generation_attempts: int = Field(
        default=2,
        ge=1,
        le=5,
        alias="AGENT_MAX_GENERATION_ATTEMPTS",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


def configure_logging(log_level: str) -> None:
    """Configure application-wide console logging."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
