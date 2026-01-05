import dspy
import pydantic
import mteval_dspy.dspy_utils
import warnings


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

    terminology: list[TerminologyEntry] = dspy.InputField(default=[])

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
            return await self.predict.aforward(*args, **kwargs)
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
}


def create_module(architecture: str) -> dspy.Module:
    assert architecture in architectures, f"Unknown architecture: {architecture}"
    return architectures[architecture]()
