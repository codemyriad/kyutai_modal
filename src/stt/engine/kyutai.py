"""Real GPU engine using the Kyutai STT model via transformers.

This module requires CUDA and should only be imported on GPU-enabled systems.
"""

from pathlib import Path

import numpy as np
import torch

from stt.constants import MODEL_ID, SAMPLE_RATE


class KyutaiEngine:
    """GPU-accelerated STT engine using kyutai/stt-1b-en_fr-trfs.

    This engine loads the Kyutai model and runs batched inference on GPU.
    It should be initialized once per container and reused for all requests.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the Kyutai engine.

        Args:
            model_path: Local path to model weights, or None to download from HuggingFace.
            device: Device to run inference on ("cuda" or "cpu").
            dtype: Model dtype (bfloat16 recommended for L40S).
        """
        self._model_path = Path(model_path) if model_path else None
        self._device = device
        self._dtype = dtype

        self._processor = None
        self._model = None

    def _load_model(self) -> None:
        """Load model and processor (lazy initialization)."""
        if self._model is not None:
            return

        from transformers import (
            KyutaiSpeechToTextForConditionalGeneration,
            KyutaiSpeechToTextProcessor,
        )

        model_id = str(self._model_path) if self._model_path else MODEL_ID

        self._processor = KyutaiSpeechToTextProcessor.from_pretrained(model_id)
        self._model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self._dtype,
            device_map="auto" if self._device == "cuda" else None,
        )

        if self._device != "cuda":
            self._model = self._model.to(self._device)

        self._model.eval()

    def transcribe_batch(self, audio_list: list[np.ndarray]) -> list[str]:
        """Transcribe a batch of audio arrays.

        Args:
            audio_list: List of float32 numpy arrays at 24kHz.

        Returns:
            List of transcription strings.
        """
        self._load_model()

        if not audio_list:
            return []

        with torch.no_grad():
            inputs = self._processor(
                audio=audio_list,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(self._model.device)

            output_ids = self._model.generate(input_values=input_values, max_new_tokens=256)
            texts = self._processor.batch_decode(output_ids, skip_special_tokens=True)

        return texts

    def warmup(self) -> None:
        """Load model and run a dummy inference to warm up CUDA kernels."""
        self._load_model()

        # Generate 1 second of dummy audio
        dummy_audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1

        with torch.no_grad():
            inputs = self._processor(
                audio=dummy_audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )
            input_values = inputs.input_values.to(self._model.device)
            _ = self._model.generate(input_values=input_values, max_new_tokens=10)

        print(f"KyutaiEngine warmed up on {self._device}")

    @property
    def device(self) -> str:
        """Return the device the model is running on."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None
