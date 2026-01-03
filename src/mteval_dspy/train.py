from pydantic import BaseModel


class LabeledFewShotConfig(BaseModel):
    pass


class SIMBAConfig(BaseModel):
    pass


class MIPROv2Config(BaseModel):
    pass


class TrainingConfig(BaseModel):
    pass


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
