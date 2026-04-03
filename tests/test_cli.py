import sys
from types import SimpleNamespace

import mteval_dspy.cli as cli_module


class _FakeSyncClient:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.calls.append(kwargs)


class _FakeAsyncClient:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.calls.append(kwargs)


def _install_fake_modules(monkeypatch):
    fake_dspy = SimpleNamespace(configure_cache=lambda **kwargs: None)
    fake_httpx = SimpleNamespace(
        Client=_FakeSyncClient,
        AsyncClient=_FakeAsyncClient,
        Limits=lambda max_connections: {"max_connections": max_connections},
    )
    fake_litellm = SimpleNamespace(client_session=None, aclient_session=None)

    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    return fake_litellm


def test_enable_ssl_verify_option_default_is_true():
    option = next(
        param for param in cli_module.cli.params if param.name == "enable_ssl_verify"
    )
    assert option.default is True


def test_cli_callback_sets_http_verify_true(monkeypatch):
    fake_litellm = _install_fake_modules(monkeypatch)
    monkeypatch.setattr(
        cli_module.mteval_dspy.dspy_utils, "setup_lm", lambda config: None
    )

    _FakeSyncClient.calls.clear()
    _FakeAsyncClient.calls.clear()

    cli_module.cli.callback(
        model="test-model",
        api_base=None,
        api_key="test-key",
        max_tokens=256,
        enable_disk_cache=False,
        max_concurrent=4,
        sampling_params="{}",
        enable_ssl_verify=True,
    )

    assert _FakeSyncClient.calls[-1]["verify"] is True
    assert _FakeAsyncClient.calls[-1]["verify"] is True
    assert fake_litellm.client_session is not None
    assert fake_litellm.aclient_session is not None


def test_cli_callback_sets_http_verify_false(monkeypatch):
    _install_fake_modules(monkeypatch)
    monkeypatch.setattr(
        cli_module.mteval_dspy.dspy_utils, "setup_lm", lambda config: None
    )

    _FakeSyncClient.calls.clear()
    _FakeAsyncClient.calls.clear()

    cli_module.cli.callback(
        model="test-model",
        api_base=None,
        api_key="test-key",
        max_tokens=256,
        enable_disk_cache=False,
        max_concurrent=4,
        sampling_params="{}",
        enable_ssl_verify=False,
    )

    assert _FakeSyncClient.calls[-1]["verify"] is False
    assert _FakeAsyncClient.calls[-1]["verify"] is False
