from pydantic import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class DBConfig(BaseConfig):
    database_url: PostgresDsn


class AppStoreClientConfig(BaseConfig):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_prefix="APPSTORE_")

    max_retries: int = 4
    retry_base_delay_seconds: float = 0.5
    throttle_delay_seconds: float = 0.3
    request_timeout_seconds: float = 10.0


class GeminiConfig(BaseConfig):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_prefix="GEMINI_")

    api_key: str = ""
    # Flash-Lite keeps per-request cost/latency low at request-serving scale.
    model_name: str = "gemini-flash-lite-latest"
    # Insight generation needs a stable model with structured-output support.
    insights_model_name: str = "gemini-3.1-flash-lite"
    # Per-attempt cap so /insights and /reports can't hang on a stuck provider request.
    insights_request_timeout_seconds: float = 30.0


db_config = DBConfig()
appstore_client_config = AppStoreClientConfig()
gemini_config = GeminiConfig()
