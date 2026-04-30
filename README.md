# mteval-dspy

mteval-dspy is a Python CLI utility for LLM-based machine translation evaluation.
It uses DSPy programs as scoring modules so you can:

- run direct scoring on JSONL inputs,
- choose among multiple scoring architectures,
- optimize prompts/programs against labeled training data.

The CLI is provider-agnostic via LiteLLM-compatible model names and OpenAI-like API endpoints.

> [!WARNING]
> This project is under heavy development. Expect rough edges, behavior changes, and occasional breaking changes between revisions.

## Highlights

- Predict DA-style scores from JSONL files or stdin.
- Train DSPy programs with MIPROv2 or SIMBA.
- Optimize against either transformed RMSE (`tRMSE`) or pairwise accuracy (`PA`).
- Optional tokenizer-based truncation of source/target segments.
- High-throughput async processing with configurable concurrency.

## Requirements

- Python 3.12+
- Access to an LLM endpoint supported by your chosen LiteLLM model string

## Installation

Install directly from GitHub (no PyPI release currently):

```bash
pip install git+https://github.com/hrabalm/mteval-dspy
```

After installation, the executable is:

```bash
mteval-dspy
```

## Quick Start (Prediction)

Input is JSONL with at least:

- `src_lang`
- `tgt_lang`
- `src`
- `tgt`

Example record:

```json
{"src_lang":"English","tgt_lang":"Czech","src":"I was not home.","tgt":"Nebyl jsem doma."}
```

Run scoring against a local OpenAI-like endpoint:

```bash
cat input.jsonl | mteval-dspy \
    --model openai/google/gemma-3-27b-it \
    --api-base http://localhost:8000 \
    --api-key NIL \
    --max-concurrent 512 \
    predict-da \
    --architecture MR7 > scored.jsonl
```

Each output line is the input object plus a predicted `score` field.

## CLI Overview

Top-level options are shared by all subcommands:

- `--model`, `-m` (required)
- `--api-base`
- `--api-key`, `-k` (defaults to `NIL`)
- `--max-tokens` (default `2048`)
- `--max-concurrent` (default `100`)
- `--sampling-params` (JSON string)
- `--enable-disk-cache / --disable-disk-cache`
- `--enable-ssl-verify / --disable-ssl-verify`
- `--http-timeout` (default `6000.0` seconds)

Supported environment variables:

- `MTEVAL_DSPY_MODEL`
- `MTEVAL_DSPY_API_BASE`
- `MTEVAL_API_KEY` or `OPENAI_API_KEY`
- `MTEVAL_DSPY_MAX_TOKENS`

## Commands

### `predict-da`

Predict scores for JSONL inputs.

Arguments/options:

- `input_file` (path or `-` for stdin; default `-`)
- `--architecture` (`DA` or `MR7`, default `DA`)
- `--trained-model`, `-m` (optional path to a saved optimized program)
- `--tokenizer` (Hugging Face tokenizer name/path)
- `--max-segment-tokens` (optional truncation limit)

Example with input file path:

```bash
mteval-dspy \
    --model openai/google/gemma-3-27b-it \
    --api-base http://localhost:8000 \
    --api-key NIL \
    predict-da data/input.jsonl \
    --architecture DA \
    --trained-model artifacts/optimized_program.json > data/scored.jsonl
```

### `train-da`

Train/optimize a scoring program and save it for later inference.

Required options:

- `--training-data` (JSONL)
- `--output`, `-o` (output path)

Important optional options:

- `--validation-data` (required by some optimizer workflows)
- `--optimizer` (`MIPROv2` or `SIMBA`, default `MIPROv2`)
- `--optimizer-params` (JSON string)
- `--optimizer-compile-params` (JSON string)
- `--objective` (`tRMSE` or `PA`, default `tRMSE`)
- `--pairwise-k-per-source` (for `PA`, default `8`)
- `--pairwise-epsilon` (for `PA`, default `0.0`)
- `--architecture` (`DA`, `MR7`, `MR7RRWA`, `MR7MEAN`; default `DA`)
- `--initial-program` (continue from a previously optimized program)
- `--training-data-max-examples`, `--validation-data-max-examples`

Example (`tRMSE` objective):

```bash
mteval-dspy \
    --model openai/google/gemma-3-27b-it \
    --api-base http://localhost:8000 \
    --api-key NIL \
    train-da \
    --training-data data/train.jsonl \
    --validation-data data/dev.jsonl \
    --architecture DA \
    --optimizer MIPROv2 \
    --objective tRMSE \
    --output artifacts/da_mipro.json
```

Example (`PA` objective):

```bash
mteval-dspy \
    --model openai/google/gemma-3-27b-it \
    --api-base http://localhost:8000 \
    --api-key NIL \
    train-da \
    --training-data data/train.jsonl \
    --validation-data data/dev.jsonl \
    --architecture MR7 \
    --optimizer SIMBA \
    --objective PA \
    --pairwise-k-per-source 8 \
    --pairwise-epsilon 1.0 \
    --output artifacts/mr7_simba_pa.json
```

## Data Formats

### Prediction input JSONL

Required fields:

- `src_lang`
- `tgt_lang`
- `src`
- `tgt`

### Training input JSONL (DA)

Required fields:

- `src_lang`
- `tgt_lang`
- `src`
- `tgt`
- `score`

Notes:

- `src_lang` and `tgt_lang` can be full language names or ISO codes (e.g. `en`, `eng`).
- Languages are normalized internally to full names where possible.

### Pairwise objective (`PA`)

For `PA`, training uses the same DA JSONL schema above, then internally samples target pairs within each `(src_lang, tgt_lang, src)` group.

## Architectures

- `DA`: Direct assessment signature; outputs a final `score` in `0-100`.
- `MR7`: Multi-criterion signature plus aggregated final `score` in `0-100`.

Other architectures are work-in-progress.

## Performance and Reliability Notes

- Increase `--max-concurrent` for throughput if your endpoint supports it.
- Use `--http-timeout` for slow backends.
- Prediction has retry logic for timeout errors and falls back to score `0` on repeated failures.
- Disk cache can be enabled with `--enable-disk-cache`.

## Development

Install dev dependencies:

```bash
uv sync --group dev
```

Run tests:

```bash
pytest
```

Optional pre-commit setup:

```bash
uv tool install pre-commit --with pre-commit-uv
pre-commit install
pre-commit run --all-files
```

## License

See [LICENSE](LICENSE).
