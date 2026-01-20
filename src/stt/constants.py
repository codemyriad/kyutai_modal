"""Core constants for the Kyutai STT service.

The kyutai/stt-1b-en_fr model uses the Mimi neural codec at 12.5 Hz (80ms frames).
Audio must be 24kHz mono PCM16.
"""

# Audio format requirements
SAMPLE_RATE: int = 24000  # Hz - required by Mimi codec
BYTES_PER_SAMPLE: int = 2  # 16-bit PCM

# Frame size: 80ms at 24kHz (minimum processing unit)
FRAME_MS: int = 80
FRAME_SAMPLES: int = 1920  # 24000 * 0.080

# Recommended streaming chunk: 480ms (6 frames)
CHUNK_MS: int = 480
CHUNK_SAMPLES: int = 11520  # 24000 * 0.480
CHUNK_BYTES: int = 23040  # 11520 * 2 bytes

# Model identification
MODEL_ID: str = "kyutai/stt-1b-en_fr-trfs"
