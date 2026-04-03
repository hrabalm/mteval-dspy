from pydantic import BaseModel, Field
from typing import Any


class DataConfig(BaseModel):
    objective: str
    trainset_path: str
    valset_path: str | None
    trainset_max_examples: int | None = None
    valset_max_examples: int | None = None


class TrainingConfig(BaseModel):
    data_config: DataConfig

    optimizer_params: dict[str, Any] = Field(default_factory=dict)
    optimizer_compile_params: dict[str, Any] = Field(default_factory=dict)


def load_trainset_valset(
    objective,
    trainset_path,
    valset_path,
    trainset_max_examples=None,
    valset_max_examples=None,
):
    import mteval_dspy.data as data

    if objective == "tRMSE":
        trainset = data.load_data_da(trainset_path, max_examples=trainset_max_examples)
        valset = data.load_data_da(valset_path, max_examples=valset_max_examples)
    elif objective == "PA":
        trainset = data.load_data_pairwise_da(
            trainset_path, max_examples=trainset_max_examples
        )
        valset = data.load_data_pairwise_da(
            valset_path, max_examples=valset_max_examples
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


def train_mipro(qe_module, config: TrainingConfig):
    import dspy

    trainset, valset = load_trainset_valset(
        objective=config.data_config.objective,
        trainset_path=config.data_config.trainset_path,
        valset_path=config.data_config.valset_path,
        trainset_max_examples=config.data_config.trainset_max_examples,
        valset_max_examples=config.data_config.valset_max_examples,
    )
    trainset = preprocess_da_dataset(
        trainset,
    )
    valset = preprocess_da_dataset(
        valset,
    )

    optimizer = dspy.MIPROv2(
        metric=tRMSE_metric,
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

    trainset, valset = load_trainset_valset(
        objective=config.data_config.objective,
        trainset_path=config.data_config.trainset_path,
        valset_path=config.data_config.valset_path,
        trainset_max_examples=config.data_config.trainset_max_examples,
        valset_max_examples=config.data_config.valset_max_examples,
    )
    trainset = preprocess_da_dataset(
        trainset,
    )
    valset = preprocess_da_dataset(
        valset,
    )
    metric = get_da_metric_from_objective(config.data_config.objective)

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


def pairwise_da_accuracy_metric(example, pred, trace=None):
    """Hard pairwise accuracy for DA scores. Expects that
    higher scores are better."""
    tgt1_score = pred.tgt1_score
    tgt2_score = pred.tgt2_score
    if example.tgt1_score > example.tgt2_score:
        return 1.0 if tgt1_score > tgt2_score else 0.0
    elif example.tgt1_score < example.tgt2_score:
        return 1.0 if tgt1_score < tgt2_score else 0.0
    else:
        return 1.0 if tgt1_score == tgt2_score else 0.0


_da_metrics_by_objective = {
    "tRMSE": tRMSE_metric,
    "PA": pairwise_da_accuracy_metric,
}


def get_da_metric_from_objective(objective: str):
    if objective in _da_metrics_by_objective:
        return _da_metrics_by_objective[objective]
    else:
        raise ValueError(f"Unknown objective: {objective}")
