from pydantic import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class DBConfig(BaseConfig):
    database_url: PostgresDsn


db_config = DBConfig()
