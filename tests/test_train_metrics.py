from types import SimpleNamespace

from mteval_dspy.train import (
    get_da_metric_from_objective,
    pairwise_da_accuracy_metric,
    tRMSE_metric,
)


def test_trmse_metric_best_score_is_one():
    example = SimpleNamespace(score=80)
    pred = SimpleNamespace(score=80)
    assert tRMSE_metric(example, pred) == 1.0


def test_trmse_metric_penalizes_difference():
    example = SimpleNamespace(score=100)
    pred = SimpleNamespace(score=0)
    assert tRMSE_metric(example, pred) == 0.0


def test_pairwise_da_accuracy_metric_correct_order():
    example = SimpleNamespace(tgt1_score=80, tgt2_score=60)
    pred = SimpleNamespace(tgt1_score=70, tgt2_score=10)
    assert pairwise_da_accuracy_metric(example, pred) == 1.0


def test_pairwise_da_accuracy_metric_incorrect_order():
    example = SimpleNamespace(tgt1_score=20, tgt2_score=60)
    pred = SimpleNamespace(tgt1_score=70, tgt2_score=10)
    assert pairwise_da_accuracy_metric(example, pred) == 0.0


def test_get_da_metric_from_objective_returns_expected_metric():
    assert get_da_metric_from_objective("tRMSE") is tRMSE_metric


def test_get_da_metric_from_objective_pa_returns_callable_with_epsilon():
    metric = get_da_metric_from_objective("PA", pairwise_epsilon=1.0)
    example = SimpleNamespace(tgt1_score=50.0, tgt2_score=49.2)
    pred = SimpleNamespace(tgt1_score=1.0, tgt2_score=99.0)
    assert metric(example, pred) == 1.0


def test_get_da_metric_from_objective_raises_for_unknown():
    try:
        get_da_metric_from_objective("UNKNOWN")
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        assert "Unknown objective" in str(exc)
