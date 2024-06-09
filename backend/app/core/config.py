from typing import Literal

from pydantic import AnyUrl

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )
    API_V1_STR: str = "/api/v1"
    DOMAIN: str = "localhost"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"

    OPENAI_API_KEY: str = "your-openai-api-key"

    BACKEND_CORS_ORIGINS: list[AnyUrl] | AnyUrl = [
        "http://localhost",
        "http://localhost:5000",
        "http://localhost:5173",
        "https://localhost",
        "https://localhost:5173",
        "https://localhost:5000",
        "http://localhost.tiangolo.com",
        "http://localhost",
        "http://localhost:5173",
        "https://localhost",
        "https://localhost:5173",
        "http://localhost.tiangolo.com",
    ]

    PROJECT_NAME: str
    CHROMA_COLLECTION: str = "langchain"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000


settings = Settings()  # type: ignore
