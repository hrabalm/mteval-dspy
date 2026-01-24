import asyncio
import json

import click
import tenacity


async def process_line(qe_module, line):
    data = json.loads(line)

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
    return output


async def process_file(qe_module, input_file, outfp):
    queue = asyncio.Queue(maxsize=2_000)

    async def producer():
        with click.open_file(input_file, "r") as f:
            for line in f:
                await queue.put(asyncio.create_task(process_line(qe_module, line)))
        await queue.put(None)  # Sentinel to indicate end of file

    async def write_worker():
        while True:
            output_item = await queue.get()
            try:
                if output_item is None:
                    break
                output = await output_item
                print(json.dumps(output, ensure_ascii=False), file=outfp)
            finally:
                queue.task_done()

    tasks = [asyncio.create_task(producer()), asyncio.create_task(write_worker())]
    await asyncio.gather(*tasks)
