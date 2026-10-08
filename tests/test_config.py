"""Settings validation: a typo in .env must fail at startup, not mid-request."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings

DB = "postgresql+asyncpg://x"


def test_thinking_level_accepts_known_values() -> None:
    assert Settings(database_url=DB, llm_thinking_level="low").llm_thinking_level == "low"


def test_empty_thinking_level_means_model_default() -> None:
    # `LLM_THINKING_LEVEL=` in .env arrives as an empty string.
    assert Settings(database_url=DB, llm_thinking_level="").llm_thinking_level is None


def test_unknown_thinking_level_is_rejected() -> None:
    with pytest.raises(ValidationError, match="LLM_THINKING_LEVEL"):
        Settings(database_url=DB, llm_thinking_level="turbo")
