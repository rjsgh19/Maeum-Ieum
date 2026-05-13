"""
애플리케이션 설정 (환경변수 → Pydantic Settings).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gcp_project_id: str
    gcp_location: str = "asia-northeast3"
    google_application_credentials: str = ""

    supabase_db_url: str
    supabase_url: str = ""
    supabase_key: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
