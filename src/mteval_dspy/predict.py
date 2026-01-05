import asyncio
import click
import json


async def process_line(qe_module, line):
    data = json.loads(line)

    prediction = await qe_module.aforward(
        **data,
    )
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
