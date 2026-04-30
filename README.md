# mteval-dspy

*mteval-dspy* is a small cli Python utility for scoring . It is built using DSPy to simplify searching for few-shot examples and to make it more independent from a particular LLM and its provider.

## Installation

The utility can be installed from its GitHub repo directly using pip. We currently don't offer PyPi packages:

```bash
pip install git+https://github.com/hrabalm/mteval-dspy
```


## Architectures

The way the particular scoring works and what its output looks like is determined by its *architecture*, which describes the outputs.

## Example usage

The input is expected in a JSONL format, where each entry has `src_lang`, `tgt_lang`, `src`, and `tgt` fields:

```json
{
    "src_lang": "English",
    "tgt_lang": "Czech",
    "src": "I was not home.",
    "tgt": "Nebyl jsem v praci.",
}
```
To score an input file input.jsonl, the utility can be used like this if we want to use the google/gemma-3-27b-it model running OpenAI-like API at [http://localhost:8000](http://localhost:8000):

```bash
cat input.jsonl | mteval-dspy \
            --model=openai/google/gemma-3-27b-it \
            --api-base http://localhost:8000 \
            --api-key=NIL \
            --max-concurrent 512 \
            predict-da \
            --architecture MR7
```
