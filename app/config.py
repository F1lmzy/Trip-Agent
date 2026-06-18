from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="nvidia/nemotron-3-ultra", alias="OPENROUTER_MODEL")
    openweather_api_key: str | None = Field(default=None, alias="OPENWEATHER_API_KEY")
    brave_search_api_key: str | None = Field(default=None, alias="BRAVE_SEARCH_API_KEY")
    chroma_path: str = Field(default="./chroma_db", alias="CHROMA_PATH")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
