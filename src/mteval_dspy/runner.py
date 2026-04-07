import asyncio
import sys
import uuid
from typing import Awaitable, Callable

import orjson
import rich
import rich.progress


class Runner:
    def __init__(
        self,
        fn: Callable[[dict], Awaitable[list[dict]]],
        /,
        reader_queue_size: int = 100,
        write_queue_size: int = 10_000,
        max_concurrent: int = 100,
    ):
        super().__init__()

        self.sleep_delta = 1e-6

        self.finished = False  # meant to be run only once

        self.fn = fn
        self.reader_queue = asyncio.Queue(maxsize=reader_queue_size)
        self.write_queue = asyncio.Queue(maxsize=write_queue_size)

        self.concurreny_semaphore = asyncio.Semaphore(max_concurrent)

    async def _reader(self, fp, task_progress):
        for line in fp:
            await self.reader_queue.put(orjson.loads(line))
            self._progress.update(task_progress, advance=1)
            await asyncio.sleep(self.sleep_delta)
        await self.reader_queue.put(None)

    async def _processor(
        self,
        task_progress_processor,
        task_progress_writer,
    ):
        total = 0
        while True:
            total += 1
            item = await self.reader_queue.get()
            if item is None:
                break

            async def wrapped_fn(*args, **kwargs):
                try:
                    kwargs.update(process_fn_idx=total - 1)
                    result = await self.fn(*args, **kwargs)
                    self._progress.update(task_progress_processor, advance=1)
                    return result
                except Exception as e:
                    print(
                        f"Warning: Error during processing: {e}, using empty result.",
                        file=sys.stderr,
                    )
                    return []
                finally:
                    self.concurreny_semaphore.release()

            await self.concurreny_semaphore.acquire()
            await self.write_queue.put(asyncio.create_task(wrapped_fn(item)))
            self._progress.update(task_progress_processor, total=total)
            self._progress.update(task_progress_writer, total=total)
            await asyncio.sleep(self.sleep_delta)
        await self.write_queue.put(None)

    async def _writer(self, fp, task_progress):
        while True:
            item = await self.write_queue.get()
            if item is None:
                break
            try:
                for result in await item:
                    fp.write(orjson.dumps(result).decode("utf-8") + "\n")
                    fp.flush()
                    # await asyncio.sleep(self.sleep_delta)
            except:
                pass  # TODO: emit warning at least
            self._progress.update(task_progress, advance=1)

    async def run(
        self,
        fp,
        out_fp,
    ):
        if self.finished:
            raise RuntimeError("Runner can only be run once.")
        with rich.progress.Progress(
            *rich.progress.Progress.get_default_columns(),
            rich.progress.TimeElapsedColumn(),
            rich.progress.MofNCompleteColumn(),
            console=rich.console.Console(file=sys.stderr),
        ) as progress:
            read_task = progress.add_task("Read", total=None)
            process_task = progress.add_task("Processed", total=None)
            write_task = progress.add_task("Written", total=None)

            try:
                self._progress = progress
                reader = self._reader(fp, read_task)
                processor = self._processor(process_task, write_task)
                writer = self._writer(out_fp, write_task)
                await asyncio.gather(reader, processor, writer)
            finally:
                self.finished = True
                del self._progress


class UnorderedRunner:
    def __init__(
        self,
        fn: Callable[[dict], Awaitable[list[dict]]],
        /,
        reader_queue_size: int = 100,
        write_queue_size: int = 10_000,
        max_concurrent: int = 100,
    ):
        super().__init__()

        self.sleep_delta = 1e-6

        self.finished = False  # meant to be run only once

        self.fn = fn
        self.reader_queue = asyncio.Queue(maxsize=reader_queue_size)
        self.concurreny_semaphore = asyncio.Semaphore(max_concurrent)
        self.tasks_to_write = asyncio.Queue(maxsize=write_queue_size)

        self.running_tasks = {}

    async def _reader(self, fp, task_progress):
        for line in fp:
            await self.reader_queue.put(orjson.loads(line))
            self._progress.update(task_progress, advance=1)
            await asyncio.sleep(self.sleep_delta)
        await self.reader_queue.put(None)

    async def _processor(
        self,
        task_progress_processor,
        task_progress_writer,
    ):
        total = 0
        while True:
            total += 1
            item = await self.reader_queue.get()
            if item is None:
                break

            async def wrapped_fn(uuid, *args, **kwargs):
                try:
                    kwargs.update(process_fn_idx=total - 1)
                    result = await self.fn(*args, **kwargs)
                    self._progress.update(task_progress_processor, advance=1)
                    await self.tasks_to_write.put(uuid)
                    return result
                finally:
                    self.concurreny_semaphore.release()

            task_uuid = str(uuid.uuid4())
            await self.concurreny_semaphore.acquire()
            self.running_tasks[task_uuid] = asyncio.create_task(
                wrapped_fn(task_uuid, item)
            )
            self._progress.update(task_progress_processor, total=total)
            self._progress.update(task_progress_writer, total=total)
            await asyncio.sleep(self.sleep_delta)
        await self.tasks_to_write.put(None)

    async def _writer(self, fp, task_progress):
        while True:
            item_uuid = await self.tasks_to_write.get()
            if item_uuid is None:
                break
            try:
                for result in await self.running_tasks.pop(item_uuid):
                    fp.write(orjson.dumps(result).decode("utf-8") + "\n")
                    fp.flush()
                    # await asyncio.sleep(self.sleep_delta)
            except:
                pass  # TODO: emit warning at least
            self._progress.update(task_progress, advance=1)

    async def run(
        self,
        fp,
        out_fp,
    ):
        if self.finished:
            raise RuntimeError("Runner can only be run once.")
        with rich.progress.Progress(
            *rich.progress.Progress.get_default_columns(),
            rich.progress.TimeElapsedColumn(),
            rich.progress.MofNCompleteColumn(),
            console=rich.console.Console(file=sys.stderr),
        ) as progress:
            read_task = progress.add_task("Read", total=None)
            process_task = progress.add_task("Processed", total=None)
            write_task = progress.add_task("Written", total=None)

            try:
                self._progress = progress
                reader = self._reader(fp, read_task)
                processor = self._processor(process_task, write_task)
                writer = self._writer(out_fp, write_task)
                await asyncio.gather(reader, processor, writer)
            finally:
                self.finished = True
                del self._progress
