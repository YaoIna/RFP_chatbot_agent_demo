from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True)

    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: SecretStr = Field(default=SecretStr(""), exclude=True, repr=False)
    llm_model: str = "deepseek-chat"
    llm_timeout_seconds: float = Field(default=180, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=2)
    rfq_database_path: Path = Path("data/rfq-agent.sqlite3")

    def safe_summary(self) -> dict[str, str | float | int]:
        return {
            "base_url": self.llm_base_url,
            "model": self.llm_model,
            "timeout_seconds": self.llm_timeout_seconds,
            "max_retries": self.llm_max_retries,
            "database_path": str(self.rfq_database_path),
        }
