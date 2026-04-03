import typing

import dspy
import pydantic
import pydantic.fields


class DSPyLMConfig(pydantic.BaseModel):
    model: str
    api_base: typing.Optional[str]
    api_key: typing.Optional[str]
    max_tokens: typing.Optional[int]
    lm_extra: dict[str, typing.Any] = pydantic.Field(default_factory=dict)


def setup_lm(lm_config: DSPyLMConfig):
    lm = dspy.LM(
        model=lm_config.model,
        api_base=lm_config.api_base,
        api_key=lm_config.api_key,
        max_tokens=lm_config.max_tokens,
        **lm_config.lm_extra,
    )
    dspy.configure(lm=lm)


class ExtendedModule(dspy.Module):
    """Extension of the dspy.Module. Added features:
    - config serialization/deserialization (config is just a plain JSON
      serializable Python dict named self.config)
    - Specification explicit specification of input/output fields of the module,
      we need this for data preparation and to give proper warnings. These
      fields also need to be serialized/deserialized.
    """

    input_fields: set[str] = {"src_lang", "tgt_lang", "src", "tgt"}
    output_fields: set[str] = {"score"}

    # Helper class to add config serialization to dspy.Module. Config is just
    # plain Python json-serializable dict
    def dump_state(self, json_mode=True):
        state = super().dump_state(json_mode=json_mode)
        if hasattr(self, "config"):
            state["config"] = self.config
        if hasattr(self, "input_fields"):
            state["input_fields"] = list(self.input_fields)
        if hasattr(self, "output_fields"):
            state["output_fields"] = list(self.output_fields)
        return state

    def load_state(self, state):
        super().load_state(state)
        if "config" in state and hasattr(self, "config"):
            self.config = state["config"]
        if "input_fields" in state and hasattr(self, "input_fields"):
            self.input_fields = set(state["input_fields"])
        if "output_fields" in state and hasattr(self, "output_fields"):
            self.output_fields = set(state["output_fields"])


class ChainOfThoughtWithPredictFallback(ExtendedModule):
    def __init__(
        self,
        signature: str | type[dspy.Signature],
        rationale_field: pydantic.fields.FieldInfo | None = None,
        rationale_field_type: type = str,
        **config: dict[str, typing.Any],
    ):
        super().__init__()


class PairwiseDA(ExtendedModule):
    """Module for converting DA to pairwise DA. This is useful for optimization/
    training."""

    input_fields: set[str] = {"src_lang", "tgt_lang", "src", "tgt1", "tgt2"}
    output_fields: set[str] = {"tgt1_score", "tgt2_score"}

    def __init__(
        self,
        module: dspy.Module,
        higher_is_better: bool = True,
    ):
        super().__init__()
        self.module = module
        self.higher_is_better = higher_is_better

    def forward(self, src_lang, tgt_lang, src, tgt1, tgt2, **kwargs):
        result1 = self.module(
            src_lang=src_lang, tgt_lang=tgt_lang, src=src, tgt=tgt1, **kwargs
        )
        result2 = self.module(
            src_lang=src_lang, tgt_lang=tgt_lang, src=src, tgt=tgt2, **kwargs
        )
        return dspy.Prediction(
            tgt1_score=result1.score,
            tgt2_score=result2.score,
        )

    async def aforward(self, src_lang, tgt_lang, src, tgt1, tgt2, **kwargs):
        result1 = self.module.acall(
            src_lang=src_lang, tgt_lang=tgt_lang, src=src, tgt=tgt1, **kwargs
        )
        result2 = self.module.acall(
            src_lang=src_lang, tgt_lang=tgt_lang, src=src, tgt=tgt2, **kwargs
        )
        return dspy.Prediction(
            tgt1_score=(await result1).score,
            tgt2_score=(await result2).score,
        )


def metric_pairwise(example, pred, trace=None):
    """DSPy metric for optimizing PairwiseDA."""
    tgt1_score = pred.tgt1_score
    tgt2_score = pred.tgt2_score
    if example.tgt1_score > example.tgt2_score:
        return 1.0 if tgt1_score > tgt2_score else 0.0
    elif example.tgt1_score < example.tgt2_score:
        return 1.0 if tgt1_score < tgt2_score else 0.0
    else:
        return 1.0 if tgt1_score == tgt2_score else 0.0
