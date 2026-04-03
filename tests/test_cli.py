import sys
from types import SimpleNamespace
from pathlib import Path

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


def test_train_da_objective_includes_pa():
    objective_option = next(
        param for param in cli_module.train_da.params if param.name == "objective"
    )
    assert "PA" in objective_option.type.choices


def test_train_da_has_pairwise_options():
    pairwise_k_option = next(
        param
        for param in cli_module.train_da.params
        if param.name == "pairwise_k_per_source"
    )
    pairwise_eps_option = next(
        param
        for param in cli_module.train_da.params
        if param.name == "pairwise_epsilon"
    )

    assert pairwise_k_option.default == 8
    assert pairwise_eps_option.default == 0.0


def test_train_da_has_initial_program_option():
    initial_program_option = next(
        param for param in cli_module.train_da.params if param.name == "initial_program"
    )
    assert initial_program_option.default is None


def test_train_da_loads_initial_program(monkeypatch, tmp_path):
    import mteval_dspy.architectures as architectures
    import mteval_dspy.train as train

    class _FakeModule:
        def __init__(self):
            self.loaded_paths = []
            self.saved_paths = []

        def load(self, path):
            self.loaded_paths.append(path)

        def save(self, path):
            self.saved_paths.append(path)

    fake_module = _FakeModule()
    monkeypatch.setattr(
        architectures, "create_module", lambda architecture: fake_module
    )
    monkeypatch.setattr(train, "train_mipro", lambda module, config: module)
    monkeypatch.setattr(train, "train_simba", lambda module, config: module)

    initial_program = tmp_path / "seed.json"
    output_path = tmp_path / "out.json"
    Path(initial_program).write_text("{}", encoding="utf-8")

    cli_module.train_da.callback(
        training_data="train.jsonl",
        training_data_max_examples=None,
        validation_data=None,
        validation_data_max_examples=None,
        optimizer="MIPROv2",
        objective="tRMSE",
        output=str(output_path),
        optimizer_params="{}",
        optimizer_compile_params="{}",
        architecture="DA",
        pairwise_k_per_source=8,
        pairwise_epsilon=0.0,
        initial_program=str(initial_program),
    )

    assert fake_module.loaded_paths == [str(initial_program)]
    assert fake_module.saved_paths == [str(output_path)]
