from typing import TYPE_CHECKING
import warnings
import random
import itertools
import math

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


def convert_example_pairwise_da(example):
    import dspy

    return dspy.Example(
        src_lang=example["src_lang"],
        tgt_lang=example["tgt_lang"],
        src=example["src"],
        tgt1=example["tgt1"],
        tgt2=example["tgt2"],
        tgt1_score=example["tgt1_score"],
        tgt2_score=example["tgt2_score"],
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
    random.seed(seed)
    random.shuffle(dataset)

    if max_examples is not None:
        dataset = dataset[:max_examples]

    return dataset


def load_data_pairwise_da(
    file_path: str,
    max_examples: int | None = None,
    seed=42,
    k_per_source: int = 8,
) -> list["dspy.Example"]:
    """Load DA JSONL data and convert it into pairwise DA examples.

    Expects src_lang, tgt_lang, src, tgt, score fields.
    Pairs are created within each (src_lang, tgt_lang, src) group.
    """
    import pandas as pd

    if k_per_source < 1:
        raise ValueError("k_per_source must be >= 1")

    df = pd.read_json(file_path, lines=True)
    required_fields = {"src_lang", "tgt_lang", "src", "tgt", "score"}
    missing_fields = sorted(required_fields - set(df.columns))
    if missing_fields:
        raise ValueError(
            "Missing required fields for DA pair conversion: "
            + ", ".join(missing_fields)
        )

    rows: list[dict] = []
    for _, row in df.iterrows():
        score = row["score"]
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError(f"Invalid score value: {score!r}")

        rows.append(
            {
                "src_lang": format_language(row["src_lang"]),
                "tgt_lang": format_language(row["tgt_lang"]),
                "src": row["src"],
                "tgt": row["tgt"],
                "score": float(score),
            }
        )

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["src_lang"], row["tgt_lang"], row["src"])
        groups.setdefault(key, []).append(row)

    rng = random.Random(seed)
    dataset = []
    for key in sorted(groups.keys()):
        candidates = groups[key]
        if len(candidates) < 2:
            continue

        pair_candidates = list(itertools.combinations(candidates, 2))
        rng.shuffle(pair_candidates)
        selected_pairs = pair_candidates[:k_per_source]

        for left, right in selected_pairs:
            example = {
                "src_lang": left["src_lang"],
                "tgt_lang": left["tgt_lang"],
                "src": left["src"],
                "tgt1": left["tgt"],
                "tgt2": right["tgt"],
                "tgt1_score": left["score"],
                "tgt2_score": right["score"],
            }
            dataset.append(convert_example_pairwise_da(example))

    rng.shuffle(dataset)
    if max_examples is not None:
        dataset = dataset[:max_examples]

    return dataset


if __name__ == "__main__":
    print(
        load_data_da(
            "/home/mhn/workspace/mteval-dspy-validation-internal/results/wmt24.encs.train.jsonl.zst",
            max_examples=2,
        )
    )

    format_language("SAFOIJASOIHF")
