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


class MR721(dspy.Signature):
    """"""

    # TODO: port from code for WMT25 paper

    ...


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
}


def create_module(architecture: str) -> dspy.Module:
    assert architecture in architectures, f"Unknown architecture: {architecture}"
    return architectures[architecture]()
