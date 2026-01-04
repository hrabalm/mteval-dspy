from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    import dspy


def set_inputs(dataset: list["dspy.Example"], input_fields: set[str]):
    return [x.with_inputs(*input_fields) for x in dataset]


def format_language(lang: str) -> str:
    """Unify language representation to full language name."""
    import iso639

    try:
        lang = iso639.Language.match(lang).name
    except iso639.language.LanguageNotFoundError:
        warnings.warn(f"Failed to match language code/language, leaving as is: {lang}")
    return lang


def convert_example_da(example):
    import dspy

    return dspy.Example(
        src_lang=example["src_lang"],
        tgt_lang=example["tgt_lang"],
        src=example["src"],
        tgt=example["tgt"],
        score=example["score"],
    )


def load_data_da(
    file_path: str, max_examples: int | None = None, seed=42
) -> list["dspy.Example"]:
    """Load JSONL training/development data for direct assessment.

    Expects src_lang, tgt_lang, src, tgt, score fields.
    Additional fields can be present and used by a specific model architecture.

    Languages can be either in ISO 639-1/3 (they will be converted to full names) or full names.
    """
    import pandas as pd

    df = pd.read_json(file_path, lines=True)
    dataset = []
    for _, row in df.iterrows():
        all_fields = row.to_dict()
        example = {
            **all_fields,
            "src_lang": format_language(row["src_lang"]),
            "tgt_lang": format_language(row["tgt_lang"]),
            "src": row["src"],
            "tgt": row["tgt"],
            "score": row["score"],
        }
        dataset.append(convert_example_da(example))
    import random

    random.seed(seed)
    random.shuffle(dataset)

    if max_examples is not None:
        dataset = dataset[:max_examples]

    return dataset


def load_data_pairwise_da(
    file_path: str, max_examples: int | None = None, seed=42
) -> list["dspy.Example"]:
    raise NotImplementedError(
        "Pairwise direct assessment data loading is not yet implemented."
    )


if __name__ == "__main__":
    print(
        load_data_da(
            "/home/mhn/workspace/mteval-dspy-validation-internal/results/wmt24.encs.train.jsonl.zst",
            max_examples=2,
        )
    )

    format_language("SAFOIJASOIHF")
