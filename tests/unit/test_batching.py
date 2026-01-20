"""Unit tests for batching logic."""

import asyncio

import numpy as np
import pytest

from stt.batching import BatchingTranscriber
from stt.engine.fake import FakeEngine


class TestFakeEngine:
    """Tests for the FakeEngine implementation."""

    def test_transcribe_single(self):
        """Single audio transcription."""
        engine = FakeEngine()
        audio = np.zeros(24000, dtype=np.float32)  # 1 second of silence
        results = engine.transcribe_batch([audio])

        assert len(results) == 1
        assert "[fake:" in results[0]
        assert "1.00s]" in results[0]

    def test_transcribe_batch_multiple(self):
        """Batch transcription of multiple audios."""
        engine = FakeEngine()
        audios = [
            np.zeros(12000, dtype=np.float32),  # 0.5s
            np.zeros(24000, dtype=np.float32),  # 1.0s
            np.zeros(36000, dtype=np.float32),  # 1.5s
        ]
        results = engine.transcribe_batch(audios)

        assert len(results) == 3
        assert "0.50s]" in results[0]
        assert "1.00s]" in results[1]
        assert "1.50s]" in results[2]

    def test_transcribe_empty_batch(self):
        """Empty batch should return empty list."""
        engine = FakeEngine()
        results = engine.transcribe_batch([])
        assert results == []

    def test_call_count(self):
        """Call count should increment."""
        engine = FakeEngine()
        assert engine.call_count == 0

        engine.transcribe_batch([np.zeros(100)])
        assert engine.call_count == 1

        engine.transcribe_batch([np.zeros(100)])
        assert engine.call_count == 2

    def test_warmup_noop(self):
        """Warmup should not raise."""
        engine = FakeEngine()
        engine.warmup()  # Should not raise

    def test_latency_simulation(self):
        """Latency simulation should delay execution."""
        import time

        engine = FakeEngine(latency_ms=50)
        start = time.time()
        engine.transcribe_batch([np.zeros(100)])
        elapsed = time.time() - start

        assert elapsed >= 0.05

    def test_deterministic_output(self):
        """Same input should produce same output."""
        engine = FakeEngine()
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        result1 = engine.transcribe_batch([audio])[0]
        result2 = engine.transcribe_batch([audio])[0]

        assert result1 == result2


class TestBatchingTranscriber:
    """Tests for the BatchingTranscriber."""

    @pytest.fixture
    def engine(self):
        return FakeEngine()

    @pytest.mark.asyncio
    async def test_single_request(self, engine):
        """Single request should be processed."""
        batcher = BatchingTranscriber(engine, max_batch_size=4, max_wait_ms=10)
        await batcher.start()

        try:
            audio = np.zeros(24000, dtype=np.float32)
            result = await batcher.transcribe(audio, "session-1")
            assert "[fake:" in result
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, engine):
        """Multiple concurrent requests should be batched."""
        batcher = BatchingTranscriber(engine, max_batch_size=4, max_wait_ms=100)
        await batcher.start()

        try:
            audios = [np.zeros(24000 * (i + 1), dtype=np.float32) for i in range(3)]
            tasks = [
                asyncio.create_task(batcher.transcribe(audio, f"session-{i}"))
                for i, audio in enumerate(audios)
            ]
            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            # All should have different durations
            assert "1.00s]" in results[0]
            assert "2.00s]" in results[1]
            assert "3.00s]" in results[2]
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_batch_size_limit(self, engine):
        """Requests should be batched up to max_batch_size."""
        batcher = BatchingTranscriber(engine, max_batch_size=2, max_wait_ms=100)
        await batcher.start()

        try:
            # Submit 4 requests - should result in 2 batches
            audios = [np.zeros(24000, dtype=np.float32) for _ in range(4)]
            tasks = [
                asyncio.create_task(batcher.transcribe(audio, f"session-{i}"))
                for i, audio in enumerate(audios)
            ]
            await asyncio.gather(*tasks)

            # Engine should have been called at least twice (2 batches of 2)
            assert engine.call_count >= 2
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_timeout_triggers_processing(self, engine):
        """Batch should be processed after max_wait_ms even if not full."""
        batcher = BatchingTranscriber(engine, max_batch_size=10, max_wait_ms=20)
        await batcher.start()

        try:
            audio = np.zeros(24000, dtype=np.float32)
            # Single request with large batch size - should still complete
            result = await asyncio.wait_for(
                batcher.transcribe(audio, "session-1"),
                timeout=1.0,
            )
            assert "[fake:" in result
        finally:
            await batcher.stop()

    @pytest.mark.asyncio
    async def test_queue_size(self, engine):
        """Queue size should reflect pending requests."""
        batcher = BatchingTranscriber(engine, max_batch_size=4, max_wait_ms=1000)
        # Don't start the batcher - requests will queue up

        batcher._running = True  # Prevent immediate processing
        batcher._queue = asyncio.Queue()

        audio = np.zeros(24000, dtype=np.float32)
        loop = asyncio.get_event_loop()

        # Manually add to queue
        await batcher._queue.put(
            __import__("stt.batching", fromlist=["BatchRequest"]).BatchRequest(
                audio, loop.create_future(), "session-1"
            )
        )

        assert batcher.queue_size == 1

    @pytest.mark.asyncio
    async def test_stop_cancels_processor(self, engine):
        """Stopping should cancel the background processor."""
        batcher = BatchingTranscriber(engine, max_batch_size=4, max_wait_ms=10)
        await batcher.start()
        assert batcher.is_running

        await batcher.stop()
        assert not batcher.is_running

    @pytest.mark.asyncio
    async def test_idempotent_start(self, engine):
        """Multiple starts should be safe."""
        batcher = BatchingTranscriber(engine, max_batch_size=4, max_wait_ms=10)
        await batcher.start()
        await batcher.start()  # Should not raise
        await batcher.stop()
