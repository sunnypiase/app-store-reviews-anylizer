"""Settings for the Gemini sentiment scoring script -- local to this eval
tooling (not app runtime config), same pydantic-settings pattern as
app.config.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class GeminiConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_prefix="GEMINI_")

    api_key: str = ""
    # gemini-flash-latest's free tier is capped at 20 requests/day (unusable at
    # this dataset's scale); flash-lite's free tier is 15 requests/minute,
    # which comfortably covers 10 concurrent requests for 1500 reviews.
    model_name: str = "gemini-flash-lite-latest"


gemini_config = GeminiConfig()
