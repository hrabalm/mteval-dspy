import pytest

import mteval_dspy.truncation as truncation_module
from mteval_dspy.truncation import truncate_segment, truncate_segment_async


def test_truncate_segment_sync():
    segment = "This is a test segment that will be truncated."
    tokenizer_name_or_path = "google/gemma-3-27b-it"
    max_tokens = 5

    truncated_segment = truncate_segment(segment, tokenizer_name_or_path, max_tokens)
    print(truncated_segment)


@pytest.mark.asyncio
async def test_truncate_segment_async():
    segment = "This is a test segment that will be truncated."
    tokenizer_name_or_path = "google/gemma-3-27b-it"
    max_tokens = 5

    truncated_segment = await truncate_segment_async(
        segment, tokenizer_name_or_path, max_tokens
    )
    print(truncated_segment)
