"""Application configuration loaded from environment variables or a .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised settings for the LUFY application.

    All values can be overridden by environment variables or a .env file.
    Field names map directly to uppercased environment variable names.
    """

    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    embed_model: str = "all-MiniLM-L6-v2"
    max_chunk_size: int = 800
    chunk_overlap: int = 150
    retrieval_top_k: int = 4
    app_title: str = "LUFY — Law Understandable For You"
    debug: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton.

    Returns:
        A fully initialised Settings instance.
    """
    return Settings()


settings: Settings = get_settings()
