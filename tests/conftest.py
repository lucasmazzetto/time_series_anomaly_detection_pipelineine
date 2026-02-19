import pytest


@pytest.fixture(autouse=True)
def _default_test_env(monkeypatch: pytest.MonkeyPatch):
    """@brief Provide stable default env values for the test suite.

    @details
    Ensures local `.env` changes do not make tests flaky. Individual tests may
    still override this value with `monkeypatch.setenv(...)` when needed.
    """
    monkeypatch.setenv("MIN_TRAINING_DATA_POINTS", "3")
    monkeypatch.setenv("LATENCY_HISTORY_LIMIT", "10")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")
