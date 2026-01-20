"""FastAPI WebSocket server for streaming STT.

This server accepts PCM16 audio over WebSocket and returns transcriptions.
It depends only on the Engine protocol, allowing use with real or fake engines.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np

from stt.audio import pcm16_to_float32, validate_audio_format
from stt.batching import BatchingTranscriber
from stt.constants import CHUNK_BYTES, CHUNK_SAMPLES, MIN_AUDIO_BYTES, MODEL_ID, SAMPLE_RATE
from stt.engine.protocol import Engine


class StreamSession:
    """Manages audio buffer state for a single WebSocket connection."""

    def __init__(self, session_id: str, chunk_threshold: int = CHUNK_BYTES):
        """Initialize a stream session.

        Args:
            session_id: Unique identifier for this session.
            chunk_threshold: Byte threshold before triggering transcription.
        """
        self.session_id = session_id
        self.chunk_threshold = chunk_threshold
        self._buffer = bytearray()

    @property
    def buffer_bytes(self) -> int:
        """Current buffer size in bytes."""
        return len(self._buffer)

    @property
    def buffer_samples(self) -> int:
        """Current buffer size in samples."""
        return len(self._buffer) // 2

    def append(self, data: bytes) -> None:
        """Append audio data to the buffer."""
        self._buffer.extend(data)

    def flush(self) -> np.ndarray:
        """Return buffer as float32 array and clear."""
        audio = pcm16_to_float32(bytes(self._buffer))
        self._buffer.clear()
        return audio

    def has_enough_data(self) -> bool:
        """Check if buffer has enough data for transcription."""
        return len(self._buffer) >= self.chunk_threshold

    def has_minimum_audio(self, min_bytes: int = MIN_AUDIO_BYTES) -> bool:
        """Check if buffer has minimum audio length for reliable transcription.

        The Kyutai model produces errors with very short audio (<1 second).
        This check prevents IndexError in model.generate() with short input.
        """
        return len(self._buffer) >= min_bytes


def create_app(engine: Engine, use_batching: bool = True) -> FastAPI:
    """Create a FastAPI application with the given engine.

    Args:
        engine: STT engine implementation (real or fake).
        use_batching: Whether to use the batching transcriber.

    Returns:
        Configured FastAPI application.
    """
    batcher: BatchingTranscriber | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal batcher
        if use_batching:
            batcher = BatchingTranscriber(engine, max_batch_size=8, max_wait_ms=50)
            await batcher.start()
        yield
        if batcher:
            await batcher.stop()

    app = FastAPI(title="Kyutai STT Service", lifespan=lifespan)

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "ok",
            "model": MODEL_ID,
            "sample_rate": SAMPLE_RATE,
            "chunk_samples": CHUNK_SAMPLES,
        }

    @app.websocket("/v1/stream")
    async def stream_transcribe(websocket: WebSocket):
        """WebSocket endpoint for streaming audio transcription.

        Protocol:
        - Client sends binary PCM16 audio chunks (24kHz mono)
        - Client sends b"EOS" to signal end of stream
        - Server responds with JSON: {"text": "...", "final": bool}
        - Server sends {"status": "complete"} when done
        """
        await websocket.accept()
        session_id = str(id(websocket))
        session = StreamSession(session_id)

        try:
            while True:
                data = await websocket.receive_bytes()

                # Handle end-of-stream signal
                if data == b"EOS":
                    # Only transcribe if we have minimum audio (prevents model errors)
                    if session.has_minimum_audio():
                        text = await _transcribe(session, batcher, engine)
                        if text and text.strip():
                            await websocket.send_json({"text": text, "final": True})
                    await websocket.send_json({"status": "complete"})
                    break

                # Validate audio format
                if not validate_audio_format(data):
                    await websocket.send_json(
                        {"error": "Invalid audio format (must be PCM16)"}
                    )
                    continue

                # Accumulate audio
                session.append(data)

                # Process when buffer reaches threshold
                if session.has_enough_data():
                    text = await _transcribe(session, batcher, engine)
                    if text and text.strip():
                        await websocket.send_json({"text": text, "final": False})

        except WebSocketDisconnect:
            pass

    return app


async def _transcribe(
    session: StreamSession,
    batcher: BatchingTranscriber | None,
    engine: Engine,
) -> str:
    """Transcribe buffered audio using batcher or direct engine call."""
    audio = session.flush()

    if batcher:
        return await batcher.transcribe(audio, session.session_id)
    else:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, engine.transcribe_batch, [audio]
        )
        return results[0] if results else ""
