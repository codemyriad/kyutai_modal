#!/usr/bin/env python3
"""
Annotate sample WAVs using OpenAI Whisper with word timestamps.

Requires:
  - OPENAI_API_KEY in environment
  - Network access to OpenAI API

Outputs (next to each WAV):
  - <file>.txt          : full transcript text
  - <file>.words.json   : list of word timestamps [{"word": "...", "start": ..., "end": ...}]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

API_URL = "https://api.openai.com/v1/audio/transcriptions"
MODEL = os.environ.get("OPENAI_MODEL", "whisper-1")
SAMPLES_DIR = Path("samples/wav24k")


def transcribe(path: Path, api_key: str) -> tuple[str, list[dict]]:
    with path.open("rb") as f:
        files = {
            "file": (path.name, f, "audio/wav"),
        }
        data = {
            "model": MODEL,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.post(API_URL, headers=headers, data=data, files=files, timeout=600)
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
    payload = resp.json()
    text = payload.get("text", "").strip()
    words: list[dict] = []
    for seg in payload.get("segments", []):
        for w in seg.get("words", []) or []:
            words.append(
                {
                    "word": w.get("word", "").strip(),
                    "start": w.get("start"),
                    "end": w.get("end"),
                }
            )
    return text, words


def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set", file=sys.stderr)
        return 1

    if not SAMPLES_DIR.exists():
        print(f"Samples dir not found: {SAMPLES_DIR}", file=sys.stderr)
        return 1

    wavs = sorted(SAMPLES_DIR.glob("*.wav"))
    if not wavs:
        print(f"No wav files found under {SAMPLES_DIR}", file=sys.stderr)
        return 1

    for wav in wavs:
        txt_path = wav.with_suffix(".wav.txt")
        words_path = wav.with_suffix(".wav.words.json")
        print(f"Transcribing {wav} -> {txt_path}, {words_path}")
        try:
            text, words = transcribe(wav, api_key)
        except Exception as exc:
            print(f"  Failed: {exc}", file=sys.stderr)
            continue

        txt_path.write_text(text, encoding="utf-8")
        words_path.write_text(json.dumps(words, indent=2), encoding="utf-8")
        print(f"  Wrote transcript ({len(text)} chars) and {len(words)} word timestamps")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
