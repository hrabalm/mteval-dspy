import asyncio
from functools import partial

import click
import tenacity

import mteval_dspy.runner
import mteval_dspy.truncation


async def process_line(
    example,
    qe_module,
    tokenizer: str,
    max_segment_tokens: int | None,
    **kwargs,
):
    data = example
    if "src" in data:
        data["src"] = await mteval_dspy.truncation.truncate_segment_async(
            data["src"], tokenizer, max_segment_tokens
        )
    if "tgt" in data:
        data["tgt"] = await mteval_dspy.truncation.truncate_segment_async(
            data["tgt"], tokenizer, max_segment_tokens
        )

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type(asyncio.TimeoutError),
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(min=1, max=10) + tenacity.wait_random(0, 2),
    )
    async def predict_with_retry(**data):
        return await qe_module.acall(
            **data,
        )

    async def predict_with_fallback(**data):
        try:
            return await predict_with_retry(**data)
        except Exception as e:
            click.echo(
                f"Warning: Error during prediction: {e}, using fallback score 0.",
                err=True,
            )
            import dspy

            return dspy.Prediction(score=0)

    prediction = await predict_with_fallback(**data)
    output = {
        **data,
        "score": prediction.score,
    }
    return [output]


async def process_file(
    qe_module,
    input_file,
    outfp,
    tokenizer: str,
    max_segment_tokens: int | None,
    max_concurrent: int,
):
    runner = mteval_dspy.runner.Runner(
        partial(
            process_line,
            qe_module=qe_module,
            tokenizer=tokenizer,
            max_segment_tokens=max_segment_tokens,
        ),
        write_queue_size=10_000,
        max_concurrent=max_concurrent,
    )
    with click.open_file(input_file, "r") as fp:
        await runner.run(fp=fp, out_fp=outfp)
