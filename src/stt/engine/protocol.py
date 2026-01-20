"""Engine protocol defining the interface for STT inference backends.

This is the "sealed boundary" that isolates GPU-dependent code from
the rest of the system (server, batching, tests).
"""

from typing import Protocol

import numpy as np


class Engine(Protocol):
    """Protocol for speech-to-text inference engines.

    Implementations must provide batched transcription and warmup methods.
    This allows swapping between real GPU engines and fake CPU engines for testing.
    """

    def transcribe_batch(self, audio_list: list[np.ndarray]) -> list[str]:
        """Transcribe a batch of audio arrays.

        Args:
            audio_list: List of float32 numpy arrays, each normalized to [-1, 1].
                       All arrays should be 24kHz mono audio.

        Returns:
            List of transcription strings, one per input audio array.
        """
        ...

    def warmup(self) -> None:
        """Warm up the engine (e.g., compile CUDA kernels, load model).

        This method is called once during container initialization and
        may be captured in a memory snapshot for faster cold starts.
        """
        ...
