"""Cross-stream request batcher for efficient GPU utilization.

Collects transcription requests from multiple concurrent WebSocket streams
and batches them together for inference.
"""

import asyncio
from dataclasses import dataclass

import numpy as np

from stt.engine.protocol import Engine


@dataclass
class BatchRequest:
    """A single transcription request waiting to be batched."""

    audio: np.ndarray
    future: asyncio.Future[str]
    session_id: str


class BatchingTranscriber:
    """Batches requests from multiple streams for efficient GPU utilization.

    This class collects transcription requests and processes them in batches,
    reducing GPU kernel launch overhead and improving throughput.
    """

    def __init__(
        self,
        engine: Engine,
        max_batch_size: int = 8,
        max_wait_ms: int = 50,
    ):
        """Initialize the batching transcriber.

        Args:
            engine: The underlying STT engine (real or fake).
            max_batch_size: Maximum number of requests per batch.
            max_wait_ms: Maximum time to wait for more requests before processing.
        """
        self._engine = engine
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue[BatchRequest] = asyncio.Queue()
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the background batch processor."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._batch_loop())

    async def stop(self) -> None:
        """Stop the background batch processor."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def transcribe(self, audio: np.ndarray, session_id: str) -> str:
        """Submit audio for batched transcription.

        Args:
            audio: Float32 audio array at 24kHz.
            session_id: Identifier for the source stream.

        Returns:
            Transcription string.
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future[str] = loop.create_future()
        await self._queue.put(BatchRequest(audio, future, session_id))
        return await future

    async def _batch_loop(self) -> None:
        """Continuously collect and process batches."""
        while self._running:
            batch: list[BatchRequest] = []

            # Wait for first request
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                batch.append(first)
            except asyncio.TimeoutError:
                continue

            # Collect more requests up to batch size or timeout
            deadline = asyncio.get_event_loop().time() + (self._max_wait_ms / 1000)

            while len(batch) < self._max_batch_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            # Process the batch
            await self._process_batch(batch)

    async def _process_batch(self, batch: list[BatchRequest]) -> None:
        """Run batched inference and resolve futures."""
        if not batch:
            return

        loop = asyncio.get_event_loop()

        def run_inference() -> list[str]:
            audio_arrays = [req.audio for req in batch]
            return self._engine.transcribe_batch(audio_arrays)

        try:
            results = await loop.run_in_executor(None, run_inference)
            for req, text in zip(batch, results):
                if not req.future.done():
                    req.future.set_result(text)
        except Exception as e:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    @property
    def queue_size(self) -> int:
        """Current number of pending requests."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        """Whether the batch processor is running."""
        return self._running
