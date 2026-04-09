import dspy
import pydantic
import mteval_dspy.dspy_utils
import warnings
from typing import Literal


class TerminologyEntry(pydantic.BaseModel):
    src: str
    tgt: str
    desc: str
    examples: list[str]


class DirectAssessmentSignature(dspy.Signature):
    """Assign score to a translation."""

    src_lang: str = dspy.InputField()
    tgt_lang: str = dspy.InputField()
    src: str = dspy.InputField()
    tgt: str = dspy.InputField()

    score: int = dspy.OutputField(desc="Score (0-100), higher is better.")


class DAWithTerminologySignature(dspy.Signature):
    """Assign score to a translation."""

    src_lang: str = dspy.InputField()
    tgt_lang: str = dspy.InputField()
    src: str = dspy.InputField()
    tgt: str = dspy.InputField()

    terminology: list[TerminologyEntry] = dspy.InputField(default_factory=list)

    score: int = dspy.OutputField(desc="Score (0-100), higher is better.")


class DAPredict(mteval_dspy.dspy_utils.ExtendedModule):
    def __init__(self, signature: type[dspy.Signature]):
        self.predict = dspy.Predict(signature=signature)

    def forward(self, *args, **kwargs):
        try:
            return self.predict(*args, **kwargs)
        except Exception as e:
            warnings.warn(f"Error during prediction: {e}, using fallback score 0.")
            return dspy.Prediction(score=0)

    async def aforward(self, *args, **kwargs):
        try:
            return await self.predict.acall(*args, **kwargs)
        except Exception as e:
            warnings.warn(
                f"Error during async prediction: {e}, using fallback score 0."
            )
            return dspy.Prediction(score=0)


# This architecture is based on our mr7.2.1 WMT25 submission
class MR7(dspy.Signature):
    src_lang: str = dspy.InputField()
    tgt_lang: str = dspy.InputField()
    src: str = dspy.InputField()
    tgt: str = dspy.InputField()

    accuracy_and_completeness_score: int = dspy.OutputField(
        desc="Accuracy and completeness score (0-10) assigned to the translation. (fidelity to the source meaning—no omissions, additions, or distortions)"
    )

    terminology_and_consistency_score: int = dspy.OutputField(
        desc="Terminology and consistency score (0-10) assigned to the translation. (domain-appropriate terms, glossary adherence, and uniformity across the text)"
    )

    fluency_and_coherence_score: int = dspy.OutputField(
        desc="Fluency and coherence score (0-10) assigned to the translation. (grammar, spelling, punctuation, smooth logical flow, and use of connectors)"
    )

    style_tone_and_audience_fit_score: int = dspy.OutputField(
        desc="Style, tone, and audience-fit score (0-10) assigned to the translation. (formality, voice, inclusivity, and suitability to the intended readership)"
    )

    locale_conventions_and_formatting_score: int = dspy.OutputField(
        desc="Locale conventions and formatting score (0-10) assigned to the translation. (numbers, units, dates, currencies, capitalization, and other locale-specific formats)"
    )

    technical_integrity_score: int = dspy.OutputField(
        desc="Technical integrity score (0-10) assigned to the translation. (placeholders, markup/tags, string length limits, and structural fidelity)"
    )

    cultural_appropriateness_score: int = dspy.OutputField(
        desc="Cultural appropriateness score (0-10) assigned to the translation. (Are idioms, references, and sensitivities suitable for the target culture?)"
    )

    score: int = dspy.OutputField(
        desc="Aggregated overall score (0-100) assigned to the translation."
    )


class MR7Predict(mteval_dspy.dspy_utils.ExtendedModule):
    """MR7 prediction with chain-of-thought and sampling-based aggregation. RRWA is inspired GEMBA-v2 by https://aclanthology.org/2025.wmt-1.67.pdf"""

    def __init__(
        self,
        num_samples: int,
        method: Literal["mean", "RRWA"],
        sampling_temperature: float = 0.5,
    ):
        super().__init__()
        self.config = {}
        self.config["num_samples"] = num_samples
        self.config["method"] = method
        self.config["sampling_temperature"] = sampling_temperature
        self.predict_no_chain_of_thought = dspy.Predict(signature=MR7)
        self.predict = dspy.ChainOfThought(signature=MR7)

    def _aggregate_scores(self, scores: list[int]) -> int:
        scores = [x for x in scores if x is not None]
        scores = [max(0, min(100, s)) for s in scores]
        if not scores:
            warnings.warn(
                "No valid scores available for aggregation; using fallback score 0."
            )
            return 0
        if self.config["method"] == "mean":
            return int(sum(scores) / len(scores))
        elif self.config["method"] == "RRWA":
            scores = sorted(scores, reverse=True)
            weights = [1 / (i + 1) for i in range(len(scores))]
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            total_weight = sum(weights)
            return round(weighted_sum / total_weight)
        else:
            raise ValueError(f"Unknown aggregation method: {self.config['method']}")

    def forward(self, *args, **kwargs):
        scores = []
        try:
            prediction = self.predict_no_chain_of_thought(
                *args,
                **kwargs,
            )
        except Exception as e:
            warnings.warn(
                f"Error during prediction without chain of thought: {e}, using fallback score 0."
            )
            prediction = dspy.Prediction(score=0)
        scores.append(prediction.score)
        for idx in range(self.config["num_samples"] - 1):
            try:
                prediction = self.predict(
                    *args,
                    **kwargs,
                    config={
                        "temperature": self.config["sampling_temperature"],
                        "rollout_id": idx,
                    },
                )
            except Exception as e:
                warnings.warn(
                    f"Error during prediction with chain of thought: {e}, using fallback score 0."
                )
                prediction = dspy.Prediction(score=0)
            scores.append(prediction.score)
        aggregated_score = self._aggregate_scores(scores)
        return dspy.Prediction(score=aggregated_score)

    async def aforward(self, *args, **kwargs):
        scores = []
        try:
            prediction = await self.predict_no_chain_of_thought.acall(
                *args,
                **kwargs,
            )
        except Exception as e:
            warnings.warn(
                f"Error during async prediction without chain of thought: {e}, using fallback score 0."
            )
            prediction = dspy.Prediction(score=0)
        scores.append(prediction.score)
        for idx in range(self.config["num_samples"] - 1):
            try:
                prediction = await self.predict.acall(
                    *args,
                    **kwargs,
                    config={
                        "temperature": self.config["sampling_temperature"],
                        "rollout_id": idx,
                    },
                )
            except Exception as e:
                warnings.warn(
                    f"Error during async prediction with chain of thought: {e}, using fallback score 0."
                )
                prediction = dspy.Prediction(score=0)
            scores.append(prediction.score)
        aggregated_score = self._aggregate_scores(scores)
        return dspy.Prediction(score=aggregated_score)

class MR8(dspy.Signature):
    src_lang: str = dspy.InputField()
    tgt_lang: str = dspy.InputField()
    src: str = dspy.InputField()
    tgt: str = dspy.InputField()

    accuracy_and_completeness_score: int = dspy.OutputField(
        desc="Accuracy and completeness score (0-10) assigned to the translation. (fidelity to the source meaning—no omissions, additions, or distortions)"
    )

    terminology_and_consistency_score: int = dspy.OutputField(
        desc="Terminology and consistency score (0-10) assigned to the translation. (domain-appropriate terms, glossary adherence, and uniformity across the text)"
    )

    fluency_and_coherence_score: int = dspy.OutputField(
        desc="Fluency and coherence score (0-10) assigned to the translation. (grammar, spelling, punctuation, smooth logical flow, and use of connectors)"
    )

    style_tone_and_audience_fit_score: int = dspy.OutputField(
        desc="Style, tone, and audience-fit score (0-10) assigned to the translation. (formality, voice, inclusivity, and suitability to the intended readership)"
    )

    locale_conventions_and_formatting_score: int = dspy.OutputField(
        desc="Locale conventions and formatting score (0-10) assigned to the translation. (numbers, units, dates, currencies, capitalization, and other locale-specific formats)"
    )

    technical_integrity_score: int = dspy.OutputField(
        desc="Technical integrity score (0-10) assigned to the translation. (placeholders, markup/tags, string length limits, and structural fidelity)"
    )

    cultural_appropriateness_score: int = dspy.OutputField(
        desc="Cultural appropriateness score (0-10) assigned to the translation. (Are idioms, references, and sensitivities suitable for the target culture?)"
    )

    score: int = dspy.OutputField(
        desc="Aggregated overall score (0-100) assigned to the translation."
    )

class ESA(dspy.Signature):
    """"""

    # TODO: port from GEMBA

    ...


class MQM(dspy.Signature):
    """"""

    # TODO: port from GEMBA

    ...


architectures = {
    "DA": lambda: DAPredict(signature=DirectAssessmentSignature),
    "MR7": lambda: DAPredict(signature=MR7),
    "MR7RRWA": lambda: MR7Predict(
        num_samples=10,
        method="RRWA",
    ),
    "MR7MEAN": lambda: MR7Predict(
        num_samples=10,
        method="mean",
    ),
}


def create_module(architecture: str) -> dspy.Module:
    if architecture not in architectures:
        available_architectures = ", ".join(sorted(architectures.keys()))
        raise ValueError(
            f"Unknown architecture: {architecture}. Available architectures: {available_architectures}"
        )
    return architectures[architecture]()
