import asyncio

from async_lru import alru_cache
from functools import cache


def load_tokenizer(tokenizer_name_or_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    return tokenizer


@alru_cache
async def load_tokenizer_async(tokenizer_name_or_path: str):
    from transformers import AutoTokenizer

    tokenizer = asyncio.to_thread(AutoTokenizer.from_pretrained, tokenizer_name_or_path)
    return await tokenizer

def truncate_segment(
    segment: str,
    tokenizer_name_or_path: str,
    max_tokens: int | None,
):
    if max_tokens is None:
        return segment

    tokenizer = load_tokenizer(tokenizer_name_or_path)
    tokens = tokenizer(segment)
    truncated_tokens = tokens.input_ids[:max_tokens]
    truncated_segment = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return truncated_segment

async def truncate_segment_async(
    segment: str,
    tokenizer_name_or_path: str,
    max_tokens: int | None,
):
    if max_tokens is None:
        return segment

    tokenizer = await load_tokenizer_async(tokenizer_name_or_path)
    tokens = tokenizer(segment)
    truncated_tokens = tokens.input_ids[:max_tokens]
    truncated_segment = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return truncated_segment
