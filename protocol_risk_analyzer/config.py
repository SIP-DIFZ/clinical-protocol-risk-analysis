from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str
    openai_base_url: str | None
    default_model: str
    openai_timeout_seconds: float
    openai_max_retries: int


def get_default_model() -> str:
    load_dotenv()
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def load_config() -> AppConfig:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required. Set it in your environment or .env file.")

    return AppConfig(
        openai_api_key=api_key,
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        default_model=get_default_model(),
        openai_timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120")),
        openai_max_retries=max(0, int(os.getenv("OPENAI_MAX_RETRIES", "2"))),
    )
