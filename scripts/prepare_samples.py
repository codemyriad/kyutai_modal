#!/usr/bin/env python3
"""Prepare audio samples for testing.

Converts audio files to 24kHz mono PCM16 WAV and generates Whisper reference transcripts.

Usage:
    uv run scripts/prepare_samples.py --in-dir raw_audio/ --out-dir samples/

Dependencies:
    uv pip install faster-whisper soundfile
    Also requires ffmpeg: sudo apt-get install ffmpeg
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sh(*cmd: str) -> None:
    """Run a shell command."""
    subprocess.check_call(cmd)


def sha256_file(p: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ffmpeg_to_wav24k(src: Path, dst: Path) -> None:
    """Convert any audio file to 24kHz mono PCM16 WAV."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    sh(
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",  # mono
        "-ar",
        "24000",  # 24kHz
        "-c:a",
        "pcm_s16le",  # 16-bit PCM
        str(dst),
    )


def duration_seconds(p: Path) -> float:
    """Get audio duration using ffprobe."""
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(p),
        ]
    )
    return float(json.loads(out)["format"]["duration"])


def whisper_transcribe(wav_path: Path, model: str) -> str:
    """Transcribe audio using faster-whisper."""
    from faster_whisper import WhisperModel

    wm = WhisperModel(model, device="cpu", compute_type="int8")
    segments, _info = wm.transcribe(str(wav_path), vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare audio samples for STT testing")
    ap.add_argument(
        "--in-dir", required=True, help="Directory with raw audio files"
    )
    ap.add_argument(
        "--out-dir", default="samples", help="Output samples directory"
    )
    ap.add_argument(
        "--whisper-model", default="large-v3", help="faster-whisper model name"
    )
    ap.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Skip Whisper transcription (useful for quick conversion)",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wav24k"
    manifest_path = out_dir / "manifest.jsonl"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # Supported audio extensions
    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".webm"}
    raw_files = [p for p in sorted(in_dir.rglob("*")) if p.suffix.lower() in exts]

    if not raw_files:
        print(f"No audio files found in {in_dir}")
        return

    print(f"Found {len(raw_files)} audio files")

    records = []
    for i, src in enumerate(raw_files, 1):
        sample_id = src.stem
        wav24k = wav_dir / f"{sample_id}.wav"

        print(f"[{i}/{len(raw_files)}] Converting {src.name}...")
        ffmpeg_to_wav24k(src, wav24k)

        ref_text = ""
        if not args.skip_whisper:
            print(f"  Transcribing with Whisper {args.whisper_model}...")
            ref_text = whisper_transcribe(wav24k, args.whisper_model)

        rec = {
            "id": sample_id,
            "wav24k_path": str(wav24k),
            "ref_text": ref_text,
            "sha256": sha256_file(wav24k),
            "duration_s": duration_seconds(wav24k),
        }
        records.append(rec)
        print(f"  Duration: {rec['duration_s']:.2f}s")

    with manifest_path.open("w", encoding="utf-8") as mf:
        for rec in records:
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {manifest_path} with {len(records)} samples")


if __name__ == "__main__":
    main()
