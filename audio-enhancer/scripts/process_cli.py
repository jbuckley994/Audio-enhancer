#!/usr/bin/env python3
"""
AudioEnhancer CLI — process audio files directly from the command line
without running the full server.

Usage:
  python scripts/process_cli.py input.wav --command "enhance distant voices"
  python scripts/process_cli.py input.mp3 --no-transcribe --output out.wav
  python scripts/process_cli.py input.wav --noise-strength 0.9 --boost-db 12
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="AudioEnhancer CLI — recover and enhance difficult audio recordings"
    )
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("--output", "-o", help="Output WAV file path (default: <input>_enhanced.wav)")
    parser.add_argument(
        "--command", "-c",
        default="enhance distant voices and reduce background noise",
        help="Natural language processing command",
    )
    parser.add_argument("--no-transcribe", action="store_true", help="Skip transcription")
    parser.add_argument("--whisper-model", default="base", choices=["tiny","base","small","medium","large"])
    parser.add_argument("--noise-strength", type=float, default=0.75, help="Noise reduction strength 0-1")
    parser.add_argument("--boost-db", type=float, default=6.0, help="Voice boost in dB")
    parser.add_argument("--chunk-seconds", type=int, default=60, help="Chunk size for processing")
    parser.add_argument("--no-source-sep", action="store_true", help="Skip source separation")
    parser.add_argument("--language", default=None, help="Language hint for Whisper (e.g. 'en', 'es')")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    import logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("cli")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_enhanced.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build config
    from app.models.schemas import PipelineConfig
    from app.core.pipeline import (
        _load_audio, _normalize, _split_chunks, _crossfade_join,
        _noise_reduce, _spectral_subtraction, _wiener_filter,
        _boost_speech_frequencies, _apply_dynamic_gain, _save_wav, _transcribe
    )

    config = PipelineConfig(
        noise_reduction_strength=args.noise_strength,
        voice_boost_db=args.boost_db,
        chunk_duration_seconds=args.chunk_seconds,
        transcribe=not args.no_transcribe,
        whisper_model=args.whisper_model,
        source_separation=not args.no_source_sep,
        language=args.language,
    )

    print(f"\n🎙  AudioEnhancer CLI")
    print(f"   Input:   {input_path}")
    print(f"   Output:  {output_path}")
    print(f"   Command: {args.command}")
    print()

    t0 = time.time()

    # Load
    print("📂 Loading audio...")
    audio, sr = _load_audio(input_path, config.target_sample_rate, config.mono)
    audio = _normalize(audio, -6.0)
    duration = len(audio) / sr
    print(f"   Duration: {duration:.1f}s  |  Sample rate: {sr}Hz")

    # Chunk
    chunks = _split_chunks(audio, sr, config.chunk_duration_seconds, config.overlap_seconds)
    print(f"   Chunks: {len(chunks)}")
    print()

    # Process
    processed = []
    for i, chunk in enumerate(chunks):
        print(f"⚙️  Chunk {i+1}/{len(chunks)}: ", end="", flush=True)
        import numpy as np
        c = chunk.copy()

        if config.noise_reduction:
            print("denoise ", end="", flush=True)
            c = _noise_reduce(c, sr, config.noise_reduction_strength, config.preserve_weak_signals)

        if config.spectral_subtraction:
            print("spectral ", end="", flush=True)
            c = _spectral_subtraction(c, sr)

        if config.wiener_filter:
            print("wiener ", end="", flush=True)
            c = _wiener_filter(c, sr)

        if config.speech_freq_boost:
            print(f"boost({config.voice_boost_db}dB) ", end="", flush=True)
            c = _boost_speech_frequencies(c, sr, config.voice_boost_db)

        if config.dynamic_gain:
            print("gain ", end="", flush=True)
            c = _apply_dynamic_gain(c, sr, config.dynamic_gain_ratio)

        peak = np.max(np.abs(c))
        if peak > 0.98:
            c = c * (0.95 / peak)

        processed.append(c)
        print("✓")

    # Reassemble
    print("\n🔗 Reassembling...")
    enhanced = _crossfade_join(processed, config.overlap_seconds, sr)
    enhanced = _normalize(enhanced, config.output_normalize_db)

    # Save
    _save_wav(output_path, enhanced, sr)
    print(f"✅ Saved: {output_path}")

    # Transcribe
    if config.transcribe:
        print(f"\n📝 Transcribing (whisper-{config.whisper_model})...")
        segments = _transcribe(output_path, config.whisper_model, config.language)

        if segments:
            transcript_path = output_path.with_suffix(".json")
            data = {
                "duration": duration,
                "segments": [s.dict() for s in segments],
                "full_text": " ".join(s.text for s in segments),
            }
            transcript_path.write_text(json.dumps(data, indent=2))

            txt_path = output_path.with_suffix(".txt")
            txt_path.write_text("\n".join(
                f"[{s.start:.1f}s – {s.end:.1f}s] {s.text}" for s in segments
            ))
            print(f"✅ Transcript: {txt_path}")
            print(f"\n--- TRANSCRIPT PREVIEW ---")
            for s in segments[:5]:
                print(f"  [{s.start:.1f}s] {s.text}")
            if len(segments) > 5:
                print(f"  ... ({len(segments)} segments total)")
        else:
            print("⚠️  No speech detected or transcription unavailable")

    elapsed = time.time() - t0
    print(f"\n⏱  Total time: {elapsed:.1f}s  (audio: {duration:.1f}s, {duration/elapsed:.1f}x realtime)")


if __name__ == "__main__":
    main()
