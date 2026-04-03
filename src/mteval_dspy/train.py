from pydantic import BaseModel, Field
from typing import Any


class DataConfig(BaseModel):
    objective: str
    trainset_path: str
    valset_path: str | None
    trainset_max_examples: int | None = None
    valset_max_examples: int | None = None
    pairwise_k_per_source: int = 8


class TrainingConfig(BaseModel):
    data_config: DataConfig

    optimizer_params: dict[str, Any] = Field(default_factory=dict)
    optimizer_compile_params: dict[str, Any] = Field(default_factory=dict)
    pairwise_epsilon: float = 0.0


def load_trainset_valset(
    objective,
    trainset_path,
    valset_path,
    trainset_max_examples=None,
    valset_max_examples=None,
    pairwise_k_per_source=8,
):
    import mteval_dspy.data as data

    if objective == "tRMSE":
        trainset = data.load_data_da(trainset_path, max_examples=trainset_max_examples)
        valset = (
            data.load_data_da(valset_path, max_examples=valset_max_examples)
            if valset_path is not None
            else None
        )
    elif objective == "PA":
        trainset = data.load_data_pairwise_da(
            trainset_path,
            max_examples=trainset_max_examples,
            k_per_source=pairwise_k_per_source,
        )
        valset = (
            data.load_data_pairwise_da(
                valset_path,
                max_examples=valset_max_examples,
                k_per_source=pairwise_k_per_source,
            )
            if valset_path is not None
            else None
        )
    else:
        raise ValueError(f"Unknown objective: {objective}")

    return trainset, valset


def preprocess_da_dataset(
    dataset,
):
    import mteval_dspy.data as data

    input_fields = {"src_lang", "tgt_lang", "src", "tgt"}

    dataset = data.set_inputs(
        dataset,
        input_fields=input_fields,
    )
    return dataset


def preprocess_pairwise_da_dataset(
    dataset,
):
    import mteval_dspy.data as data

    input_fields = {"src_lang", "tgt_lang", "src", "tgt1", "tgt2"}

    dataset = data.set_inputs(
        dataset,
        input_fields=input_fields,
    )
    return dataset


def get_preprocess_function_for_objective(objective: str):
    if objective == "tRMSE":
        return preprocess_da_dataset
    if objective == "PA":
        return preprocess_pairwise_da_dataset
    raise ValueError(f"Unknown objective: {objective}")


def train_mipro(qe_module, config: TrainingConfig):
    import dspy
    import mteval_dspy.dspy_utils

    if config.data_config.objective == "PA":
        qe_module = mteval_dspy.dspy_utils.PairwiseDA(qe_module)

    preprocess_fn = get_preprocess_function_for_objective(config.data_config.objective)
    metric = get_da_metric_from_objective(
        config.data_config.objective,
        pairwise_epsilon=config.pairwise_epsilon,
    )

    trainset, valset = load_trainset_valset(
        objective=config.data_config.objective,
        trainset_path=config.data_config.trainset_path,
        valset_path=config.data_config.valset_path,
        trainset_max_examples=config.data_config.trainset_max_examples,
        valset_max_examples=config.data_config.valset_max_examples,
        pairwise_k_per_source=config.data_config.pairwise_k_per_source,
    )
    trainset = preprocess_fn(
        trainset,
    )
    valset = preprocess_fn(valset) if valset is not None else None

    optimizer = dspy.MIPROv2(
        metric=metric,
        **config.optimizer_params,
    )

    optimized_program = optimizer.compile(
        qe_module,
        trainset=trainset,
        valset=valset,
        requires_permission_to_run=False,
        **config.optimizer_compile_params,
    )
    return optimized_program


def train_simba(qe_module, config: TrainingConfig):
    import dspy
    import mteval_dspy.dspy_utils

    if config.data_config.objective == "PA":
        qe_module = mteval_dspy.dspy_utils.PairwiseDA(qe_module)

    preprocess_fn = get_preprocess_function_for_objective(config.data_config.objective)

    trainset, valset = load_trainset_valset(
        objective=config.data_config.objective,
        trainset_path=config.data_config.trainset_path,
        valset_path=config.data_config.valset_path,
        trainset_max_examples=config.data_config.trainset_max_examples,
        valset_max_examples=config.data_config.valset_max_examples,
        pairwise_k_per_source=config.data_config.pairwise_k_per_source,
    )
    trainset = preprocess_fn(
        trainset,
    )
    metric = get_da_metric_from_objective(
        config.data_config.objective,
        pairwise_epsilon=config.pairwise_epsilon,
    )

    optimizer = dspy.SIMBA(
        metric=metric,
        **config.optimizer_params,
    )

    optimized_program = optimizer.compile(
        qe_module,
        trainset=trainset,
        **config.optimizer_compile_params,
    )
    return optimized_program


def tRMSE_metric(example, pred, trace=None):
    """RMSE transformed into 0-1 range, higher is better
    # expects scores in 0-100 range"""
    return 1 - ((example.score - pred.score) / 100) ** 2


def pairwise_da_accuracy_metric(example, pred, trace=None, epsilon: float = 0.0):
    """Hard pairwise accuracy for DA scores. Expects that
    higher scores are better."""
    gold_delta = example.tgt1_score - example.tgt2_score
    if abs(gold_delta) <= epsilon:
        return 1.0

    pred_delta = pred.tgt1_score - pred.tgt2_score
    return 1.0 if (gold_delta > 0) == (pred_delta > 0) else 0.0


_da_metrics_by_objective = {
    "tRMSE": tRMSE_metric,
    "PA": pairwise_da_accuracy_metric,
}


def get_da_metric_from_objective(objective: str, pairwise_epsilon: float = 0.0):
    if objective == "PA":
        return lambda example, pred, trace=None: pairwise_da_accuracy_metric(
            example,
            pred,
            trace=trace,
            epsilon=pairwise_epsilon,
        )
    if objective in _da_metrics_by_objective:
        return _da_metrics_by_objective[objective]
    else:
        raise ValueError(f"Unknown objective: {objective}")
