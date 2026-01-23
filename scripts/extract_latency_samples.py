#!/usr/bin/env python3
"""Extract sentence-aligned clips from test_audio.wav for latency testing.

This script uses the timestamp-annotated transcript in test_audio.json to
carve out sentence-level examples (default: 5 clips, each <= 30s) and writes
matching .wav/.txt/.words.json files into samples/wav24k so latency_test.py
can exercise multiple varied inputs.
"""

from __future__ import annotations

import argparse
import json
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


TOKEN_PATTERN = r"\b\w+['\w]*\b"


def _normalize(token: str) -> str:
    """Lowercase and strip leading/trailing punctuation/currency."""
    cleaned = re.sub(r"^[^0-9A-Za-z]+|[^0-9A-Za-z]+$", "", token)
    return cleaned.lower()


@dataclass
class SentenceSpan:
    text: str
    start_idx: int
    end_idx: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def load_transcript(transcript_path: Path) -> tuple[list[SentenceSpan], list[dict], float]:
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    raw_words = [w for w in data["words"] if str(w.get("word", "")).strip()]
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", data["text"].strip())
        if s.strip()
    ]

    token_count = sum(len(re.findall(TOKEN_PATTERN, s)) for s in sentences)
    if token_count != len(raw_words):
        raise ValueError(f"Token/word count mismatch: {token_count} vs {len(raw_words)}")

    spans: list[SentenceSpan] = []
    cursor = 0
    for sentence in sentences:
        tokens = re.findall(TOKEN_PATTERN, sentence)
        if not tokens:
            continue
        start_idx = cursor
        end_idx = cursor + len(tokens) - 1
        # Sanity check alignment against the word timeline
        for offset, tok in enumerate(tokens):
            word = raw_words[start_idx + offset]["word"]
            if _normalize(tok) != _normalize(word):
                raise ValueError(
                    f"Token mismatch at word {start_idx + offset}: '{tok}' vs '{word}'"
                )
        spans.append(
            SentenceSpan(
                text=sentence,
                start_idx=start_idx,
                end_idx=end_idx,
                start=float(raw_words[start_idx]["start"]),
                end=float(raw_words[end_idx]["end"]),
            )
        )
        cursor += len(tokens)

    if cursor != len(raw_words):
        raise ValueError(f"Cursor ended at {cursor}, expected {len(raw_words)}")

    return spans, raw_words, float(data["duration"])


def choose_examples(
    spans: Iterable[SentenceSpan],
    total_duration: float,
    count: int,
    max_duration: float,
    min_duration: float,
) -> list[SentenceSpan]:
    candidates = [s for s in spans if min_duration <= s.duration <= max_duration]
    if len(candidates) < count:
        # Relax minimum duration if we don't have enough options.
        candidates = [s for s in spans if s.duration <= max_duration]

    anchors = [(total_duration * (i + 0.5) / count) for i in range(count)]
    remaining = candidates.copy()
    selected: list[SentenceSpan] = []

    for anchor in anchors:
        if not remaining:
            break
        remaining.sort(key=lambda s: (abs(s.start - anchor), -s.duration))
        selected.append(remaining.pop(0))

    if len(selected) < count:
        extras = [s for s in spans if s.duration <= max_duration and s not in selected]
        extras.sort(key=lambda s: s.duration, reverse=True)
        for extra in extras:
            if len(selected) >= count:
                break
            selected.append(extra)

    selected.sort(key=lambda s: s.start)
    return selected[:count]


def read_wav(path: Path) -> tuple[np.ndarray, int, int, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    if width not in dtype_map:
        raise ValueError(f"Unsupported sample width: {width} bytes")

    audio = np.frombuffer(raw, dtype=dtype_map[width])
    audio = audio.reshape(-1, channels)
    return audio, rate, width, channels


def write_wav(path: Path, audio: np.ndarray, rate: int, width: int, channels: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())


def extract_clips(
    audio_path: Path,
    transcript_path: Path,
    out_dir: Path,
    prefix: str,
    count: int,
    max_duration: float,
    min_duration: float,
) -> list[dict]:
    spans, words, total_duration = load_transcript(transcript_path)
    chosen = choose_examples(spans, total_duration, count, max_duration, min_duration)
    audio, rate, width, channels = read_wav(audio_path)
    total_frames = audio.shape[0]

    manifest: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, span in enumerate(chosen, start=1):
        start_sample = max(0, int(round(span.start * rate)))
        end_sample = min(total_frames, int(round(span.end * rate)))
        if end_sample <= start_sample:
            continue

        clip = np.copy(audio[start_sample:end_sample])
        base_name = f"{prefix}_{idx:02d}"
        wav_path = out_dir / f"{base_name}.wav"
        txt_path = out_dir / f"{base_name}.wav.txt"
        words_path = out_dir / f"{base_name}.wav.words.json"

        write_wav(wav_path, clip, rate, width, channels)
        txt_path.write_text(span.text + "\n", encoding="utf-8")

        clip_words = []
        for word in words[span.start_idx : span.end_idx + 1]:
            clip_words.append(
                {
                    "word": word["word"],
                    "start": float(word["start"]) - span.start,
                    "end": float(word["end"]) - span.start,
                }
            )
        words_path.write_text(json.dumps(clip_words, indent=2), encoding="utf-8")

        manifest.append(
            {
                "id": base_name,
                "text": span.text,
                "start": span.start,
                "end": span.end,
                "duration": span.duration,
                "paths": {
                    "wav": str(wav_path),
                    "txt": str(txt_path),
                    "words": str(words_path),
                },
            }
        )

    manifest_path = out_dir / f"{prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract <=30s sentence clips from test_audio.wav for latency testing."
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("test_audio.wav"),
        help="Source WAV file (default: test_audio.wav)",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=Path("test_audio.json"),
        help="Timestamped transcript JSON (default: test_audio.json)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("samples/wav24k"),
        help="Directory for extracted clips (default: samples/wav24k)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="latency_example",
        help="Filename prefix for generated clips",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of examples to extract",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum clip duration in seconds",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=6.0,
        help="Minimum clip duration in seconds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = extract_clips(
        audio_path=args.audio,
        transcript_path=args.transcript,
        out_dir=args.out_dir,
        prefix=args.prefix,
        count=args.count,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
    )
    print(f"Extracted {len(manifest)} clips to {args.out_dir}")
    for item in manifest:
        print(f"- {item['id']}: {item['duration']:.2f}s @ {item['start']:.2f}s")


if __name__ == "__main__":
    main()
