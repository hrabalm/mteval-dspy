import json
from pathlib import Path

import pytest

from mteval_dspy.data import load_data_pairwise_da


def _write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")


def test_load_data_pairwise_da_converts_within_source_and_samples(tmp_path):
    data_file = tmp_path / "da.jsonl"
    rows = [
        {
            "src_lang": "en",
            "tgt_lang": "de",
            "src": "Hello",
            "tgt": "Hallo",
            "score": 90,
        },
        {
            "src_lang": "en",
            "tgt_lang": "de",
            "src": "Hello",
            "tgt": "Servus",
            "score": 80,
        },
        {
            "src_lang": "en",
            "tgt_lang": "de",
            "src": "Hello",
            "tgt": "Hi",
            "score": 70,
        },
        {
            "src_lang": "en",
            "tgt_lang": "de",
            "src": "World",
            "tgt": "Welt",
            "score": 95,
        },
        {
            "src_lang": "en",
            "tgt_lang": "de",
            "src": "World",
            "tgt": "Erde",
            "score": 60,
        },
    ]
    _write_jsonl(data_file, rows)

    dataset = load_data_pairwise_da(str(data_file), k_per_source=2, seed=7)

    # "Hello" has 3 candidates -> 2 sampled pairs, "World" has 2 -> 1 pair.
    assert len(dataset) == 3
    for example in dataset:
        assert hasattr(example, "tgt1")
        assert hasattr(example, "tgt2")
        assert hasattr(example, "tgt1_score")
        assert hasattr(example, "tgt2_score")


def test_load_data_pairwise_da_validates_required_columns(tmp_path):
    data_file = tmp_path / "bad.jsonl"
    _write_jsonl(
        data_file,
        [{"src_lang": "en", "tgt_lang": "de", "src": "x", "tgt": "y"}],
    )

    with pytest.raises(ValueError, match="Missing required fields"):
        load_data_pairwise_da(str(data_file))


def test_load_data_pairwise_da_rejects_invalid_score(tmp_path):
    data_file = tmp_path / "bad-score.jsonl"
    _write_jsonl(
        data_file,
        [
            {
                "src_lang": "en",
                "tgt_lang": "de",
                "src": "x",
                "tgt": "y",
                "score": "not-a-number",
            }
        ],
    )

    with pytest.raises(ValueError, match="Invalid score value"):
        load_data_pairwise_da(str(data_file))


def test_load_data_pairwise_da_max_examples_cap(tmp_path):
    data_file = tmp_path / "da.jsonl"
    rows = [
        {
            "src_lang": "en",
            "tgt_lang": "de",
            "src": "Hello",
            "tgt": "Hallo",
            "score": 90,
        },
        {
            "src_lang": "en",
            "tgt_lang": "de",
            "src": "Hello",
            "tgt": "Servus",
            "score": 80,
        },
        {
            "src_lang": "en",
            "tgt_lang": "de",
            "src": "Hello",
            "tgt": "Hi",
            "score": 70,
        },
    ]
    _write_jsonl(data_file, rows)

    dataset = load_data_pairwise_da(str(data_file), max_examples=1, k_per_source=5)
    assert len(dataset) == 1
