from protocol_risk_analyzer.config import get_default_model


def test_get_default_model_returns_fallback(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert get_default_model() == "gpt-4.1-mini"


def test_get_default_model_returns_env_value(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5")
    assert get_default_model() == "gpt-5"
