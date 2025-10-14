from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings."""

    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI Teacher Backend"

    # CORS
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o"

    # Course Settings
    COURSE_TOPIC: str = "Python Programming"

    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_SECOND: float = 2.0
    RATE_LIMIT_CHECK_INTERVAL: float = 0.1
    RATE_LIMIT_MAX_BURST: int = 10

    # Retry Policy
    RETRY_MAX_ATTEMPTS: int = 3
    RETRY_INITIAL_INTERVAL: float = 1.0
    RETRY_BACKOFF_FACTOR: float = 2.0
    RETRY_MAX_INTERVAL: float = 10.0

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
