import sys
from types import SimpleNamespace

import mteval_dspy.dspy_utils as dspy_utils
import mteval_dspy.train as train


def test_pairwise_metric_with_epsilon_gold_tie_is_correct():
    example = SimpleNamespace(tgt1_score=81.0, tgt2_score=80.5)
    pred = SimpleNamespace(tgt1_score=10.0, tgt2_score=90.0)
    assert train.pairwise_da_accuracy_metric(example, pred, epsilon=1.0) == 1.0


def test_pairwise_metric_without_epsilon_requires_direction():
    example = SimpleNamespace(tgt1_score=81.0, tgt2_score=80.5)
    pred = SimpleNamespace(tgt1_score=10.0, tgt2_score=90.0)
    assert train.pairwise_da_accuracy_metric(example, pred, epsilon=0.0) == 0.0


def test_preprocess_pairwise_da_dataset_sets_inputs():
    class _Example:
        def __init__(self):
            self.inputs = None

        def with_inputs(self, *fields):
            self.inputs = set(fields)
            return self

    dataset = [_Example()]
    out = train.preprocess_pairwise_da_dataset(dataset)
    assert out[0].inputs == {"src_lang", "tgt_lang", "src", "tgt1", "tgt2"}


def test_train_mipro_wraps_module_for_pa(monkeypatch):
    class _Example:
        def with_inputs(self, *fields):
            return self

    class _FakeMIPROv2:
        last_module = None

        def __init__(self, metric, **kwargs):
            self.metric = metric

        def compile(
            self, module, trainset, valset, requires_permission_to_run, **kwargs
        ):
            self.__class__.last_module = module
            return module

    fake_dspy = SimpleNamespace(MIPROv2=_FakeMIPROv2)
    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)

    monkeypatch.setattr(
        train,
        "load_trainset_valset",
        lambda **kwargs: ([_Example()], [_Example()]),
    )

    config = train.TrainingConfig(
        data_config=train.DataConfig(
            objective="PA",
            trainset_path="train.jsonl",
            valset_path="val.jsonl",
            pairwise_k_per_source=2,
        ),
        pairwise_epsilon=0.25,
    )

    module = object()
    out = train.train_mipro(module, config)

    assert isinstance(_FakeMIPROv2.last_module, dspy_utils.PairwiseDA)
    assert out is _FakeMIPROv2.last_module
