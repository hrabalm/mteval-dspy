import dspy

import typing
import pydantic
import pydantic.fields


class DSPyLMConfig(pydantic.BaseModel):
    pass


def setup_lm(lm_config: DSPyLMConfig):
    pass


class ChainOfThoughtWithPredictFallback(dspy.Module):
    def __init__(
        self,
        signature: str | type[dspy.Signature],
        rationale_field: pydantic.fields.FieldInfo | None = None,
        rationale_field_type: type = str,
        **config: dict[str, typing.Any],
    ):
        super().__init__()
