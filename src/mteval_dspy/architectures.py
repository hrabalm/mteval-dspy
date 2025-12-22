import dspy
import pydantic


class TerminologyEntry(pydantic.BaseModel):
    src: str
    tgt: str
    desc: str
    examples: list[str]


class DA(dspy.Signature):
    """Assign score to a translation."""

    src_lang: str = dspy.InputField()
    tgt_lang: str = dspy.InputField()
    src: str = dspy.InputField()
    tgt: str = dspy.InputField()

    terminology: list[TerminologyEntry] = dspy.InputField(default=[])

    score: int = dspy.InputField(desc="Score (0-100), higher is better.")


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


def create_module() -> dspy.Module:
    pass
