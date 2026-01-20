"""GPU integration tests for the Kyutai engine.

These tests require a GPU and should be run on Modal.

Run with: uvx modal run tests/integration_gpu/test_engine.py
"""

import numpy as np
import pytest

# Mark all tests in this module as requiring GPU
pytestmark = pytest.mark.gpu


class TestKyutaiEngine:
    """Integration tests for KyutaiEngine on GPU."""

    @pytest.mark.skip(reason="Requires GPU - run on Modal")
    def test_engine_loads(self):
        """Engine should load model successfully."""
        from stt.engine.kyutai import KyutaiEngine

        engine = KyutaiEngine(device="cuda")
        engine.warmup()
        assert engine.is_loaded

    @pytest.mark.skip(reason="Requires GPU - run on Modal")
    def test_engine_transcribes(self):
        """Engine should produce transcription."""
        from stt.engine.kyutai import KyutaiEngine

        engine = KyutaiEngine(device="cuda")
        engine.warmup()

        # 1 second of silence
        audio = np.zeros(24000, dtype=np.float32)
        results = engine.transcribe_batch([audio])

        assert len(results) == 1
        assert isinstance(results[0], str)

    @pytest.mark.skip(reason="Requires GPU - run on Modal")
    def test_engine_batch_inference(self):
        """Engine should handle batch inference."""
        from stt.engine.kyutai import KyutaiEngine

        engine = KyutaiEngine(device="cuda")
        engine.warmup()

        audios = [
            np.random.randn(24000).astype(np.float32) * 0.1,
            np.random.randn(48000).astype(np.float32) * 0.1,
        ]
        results = engine.transcribe_batch(audios)

        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)


# Modal-specific test runner
if __name__ == "__main__":
    import modal

    app = modal.App("kyutai-stt-gpu-tests")

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.4.0",
            "transformers>=4.53.0",
            "accelerate>=0.33.0",
            "huggingface-hub[hf_transfer]>=0.25.0",
            "numpy<2",
            "pytest>=8.0.0",
        )
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    )

    @app.function(gpu="L40S", image=image, timeout=600)
    def run_gpu_tests():
        """Run GPU tests on Modal."""
        from stt.engine.kyutai import KyutaiEngine

        print("Loading engine...")
        engine = KyutaiEngine(device="cuda")
        engine.warmup()
        print(f"Engine loaded on {engine.device}")

        # Test 1: Simple transcription
        print("\nTest 1: Simple transcription")
        audio = np.zeros(24000, dtype=np.float32)
        results = engine.transcribe_batch([audio])
        print(f"  Result: {results}")
        assert len(results) == 1

        # Test 2: Batch inference
        print("\nTest 2: Batch inference")
        audios = [
            np.random.randn(24000).astype(np.float32) * 0.1,
            np.random.randn(48000).astype(np.float32) * 0.1,
            np.random.randn(36000).astype(np.float32) * 0.1,
        ]
        results = engine.transcribe_batch(audios)
        print(f"  Results: {results}")
        assert len(results) == 3

        print("\nAll GPU tests passed!")
        return "success"

    @app.local_entrypoint()
    def main():
        result = run_gpu_tests.remote()
        print(f"Remote result: {result}")
