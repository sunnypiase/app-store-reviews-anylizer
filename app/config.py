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


db_config = DBConfig()
appstore_client_config = AppStoreClientConfig()
