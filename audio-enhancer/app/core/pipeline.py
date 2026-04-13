"""
AudioPipeline — the core processing engine.

Stages:
  1. Preprocessing    — load, convert, normalize
  2. Chunking         — split into manageable segments
  3. VAD              — detect speech regions
  4. Source Separation — isolate voice layer
  5. Noise Reduction  — remove noise while preserving faint speech
  6. Voice Enhancement — boost distant/faint speech
  7. Spectral Processing — Wiener filter + spectral subtraction
  8. Post-processing  — reassemble, normalize, export
  9. Transcription    — optional Whisper pass
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from app.core.job_manager import JobManager
from app.models.schemas import PipelineConfig, TranscriptSegment
from app.utils.file_handler import FileHandler
from app.utils.model_loader import ModelLoader

logger = logging.getLogger("audio_enhancer.pipeline")
file_handler = FileHandler()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_audio(path: Path, target_sr: int = 16000, mono: bool = True) -> Tuple[np.ndarray, int]:
    """Load audio file → (samples float32, sample_rate)."""
    import librosa
    audio, sr = librosa.load(str(path), sr=target_sr, mono=mono)
    logger.info(f"Loaded {path.name}: {audio.shape}, sr={sr}, duration={len(audio)/sr:.1f}s")
    return audio.astype(np.float32), sr


def _save_wav(path: Path, audio: np.ndarray, sr: int):
    """Save float32 numpy array as WAV."""
    import soundfile as sf
    sf.write(str(path), audio, sr, subtype="PCM_16")


def _normalize(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Peak-normalize to target dBFS."""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    target_amp = 10 ** (target_db / 20.0)
    return audio * (target_amp / peak)


def _split_chunks(audio: np.ndarray, sr: int, chunk_sec: int, overlap_sec: int) -> List[np.ndarray]:
    """Split audio into overlapping chunks."""
    chunk_samples = chunk_sec * sr
    overlap_samples = overlap_sec * sr
    step = chunk_samples - overlap_samples
    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + chunk_samples, len(audio))
        chunks.append(audio[start:end])
        if end == len(audio):
            break
        start += step
    return chunks


def _crossfade_join(chunks: List[np.ndarray], overlap_sec: int, sr: int) -> np.ndarray:
    """Reassemble chunks with cosine crossfade in overlap regions."""
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    overlap = overlap_sec * sr
    result = chunks[0].copy()

    for chunk in chunks[1:]:
        if overlap == 0 or len(result) < overlap or len(chunk) < overlap:
            result = np.concatenate([result, chunk])
            continue

        # Cosine fade window
        t = np.linspace(0, np.pi / 2, overlap)
        fade_out = np.cos(t) ** 2
        fade_in = np.sin(t) ** 2

        blended = result[-overlap:] * fade_out + chunk[:overlap] * fade_in
        result = np.concatenate([result[:-overlap], blended, chunk[overlap:]])

    return result


# ─── Noise reduction ──────────────────────────────────────────────────────────

def _noise_reduce(audio: np.ndarray, sr: int, strength: float, preserve_weak: bool) -> np.ndarray:
    """
    Multi-stage noise reduction:
    1. noisereduce spectral gating
    2. Optional Wiener smoothing
    Strength 0-1 controls aggressiveness; preserve_weak protects faint signals.
    """
    try:
        import noisereduce as nr

        # Estimate noise from first 0.5s (or wherever it's quietest)
        noise_sample_len = min(int(sr * 0.5), len(audio) // 4)
        noise_clip = audio[:noise_sample_len]

        # Adjust prop_decrease based on strength; when preserve_weak, cap it
        prop = min(strength, 0.9) if preserve_weak else strength

        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            y_noise=noise_clip,
            prop_decrease=prop,
            stationary=False,
            n_fft=2048,
            win_length=None,
            hop_length=512,
            n_std_thresh_stationary=1.5,
            time_constant_s=2.0,
        )
        return reduced.astype(np.float32)
    except ImportError:
        logger.warning("noisereduce not installed — skipping noise reduction")
        return audio
    except Exception as e:
        logger.error(f"Noise reduction failed: {e}")
        return audio


# ─── Spectral subtraction ─────────────────────────────────────────────────────

def _spectral_subtraction(audio: np.ndarray, sr: int, alpha: float = 2.0) -> np.ndarray:
    """
    Classic spectral subtraction with half-wave rectification.
    alpha: over-subtraction factor (higher = more aggressive)
    """
    import librosa

    n_fft = 2048
    hop = 512
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    magnitude, phase = np.abs(stft), np.angle(stft)

    # Noise estimate: minimum statistics across time
    noise_est = np.min(magnitude, axis=1, keepdims=True) * 1.5

    # Subtract and half-wave rectify (floor at small beta)
    beta = 0.01
    enhanced_mag = np.maximum(magnitude - alpha * noise_est, beta * magnitude)

    # Reconstruct
    enhanced_stft = enhanced_mag * np.exp(1j * phase)
    enhanced = librosa.istft(enhanced_stft, hop_length=hop, length=len(audio))
    return enhanced.astype(np.float32)


# ─── Wiener filter ────────────────────────────────────────────────────────────

def _wiener_filter(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Frequency-domain Wiener filter.
    Estimates signal PSD and noise PSD, computes Wiener gain.
    """
    try:
        import librosa

        n_fft = 2048
        hop = 512
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
        power = np.abs(stft) ** 2

        # Noise power: minimum of each frequency bin across time
        noise_power = np.min(power, axis=1, keepdims=True)
        noise_power = np.maximum(noise_power, 1e-10)

        # Wiener gain G = max(SNR / (SNR+1), floor)
        snr = np.maximum((power - noise_power) / noise_power, 0)
        gain = snr / (snr + 1.0)
        gain = np.maximum(gain, 0.05)  # floor to prevent full nulling

        filtered_stft = stft * gain
        filtered = librosa.istft(filtered_stft, hop_length=hop, length=len(audio))
        return filtered.astype(np.float32)
    except Exception as e:
        logger.error(f"Wiener filter failed: {e}")
        return audio


# ─── Speech frequency boost ───────────────────────────────────────────────────

def _boost_speech_frequencies(audio: np.ndarray, sr: int, boost_db: float = 6.0) -> np.ndarray:
    """
    Boost the human speech band (300–3400 Hz) using a bandpass EQ.
    Also applies a gentle presence boost at 2–5 kHz for clarity.
    """
    try:
        from scipy.signal import butter, filtfilt, sosfilt

        # Convert dB to linear gain
        gain = 10 ** (boost_db / 20.0)

        # Design bandpass for speech fundamentals 300-3400 Hz
        low = 300 / (sr / 2)
        high = min(3400 / (sr / 2), 0.99)
        if low >= high:
            return audio

        sos = butter(4, [low, high], btype="bandpass", output="sos")
        speech_band = sosfilt(sos, audio)

        # Mix boosted band back
        enhanced = audio + speech_band * (gain - 1.0)

        # Presence boost 2k-5k for intelligibility
        low2 = min(2000 / (sr / 2), 0.99)
        high2 = min(5000 / (sr / 2), 0.99)
        if low2 < high2:
            sos2 = butter(2, [low2, high2], btype="bandpass", output="sos")
            presence = sosfilt(sos2, audio)
            enhanced = enhanced + presence * 0.5

        # Prevent clipping
        peak = np.max(np.abs(enhanced))
        if peak > 1.0:
            enhanced = enhanced / peak * 0.95

        return enhanced.astype(np.float32)
    except ImportError:
        logger.warning("scipy not installed — skipping frequency boost")
        return audio


# ─── Dynamic gain (upward expansion / compression) ────────────────────────────

def _apply_dynamic_gain(
    audio: np.ndarray, sr: int, ratio: float = 3.0, threshold_db: float = -40.0
) -> np.ndarray:
    """
    Upward expander: amplifies quiet sections (faint voices) more than loud ones.
    Works on short frames to track envelope.
    """
    frame_len = int(sr * 0.02)  # 20ms frames
    hop = frame_len // 2
    output = np.zeros_like(audio)

    threshold_amp = 10 ** (threshold_db / 20.0)

    for i in range(0, len(audio) - frame_len, hop):
        frame = audio[i : i + frame_len]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-12)

        if rms < threshold_amp:
            # Boost quiet frames
            gain = (threshold_amp / (rms + 1e-12)) ** (1 - 1 / ratio)
            gain = min(gain, 10.0)  # cap at 20dB boost
        else:
            gain = 1.0

        output[i : i + frame_len] += frame * gain * 0.5  # overlap-add

    # Handle last chunk
    if len(audio) % hop != 0:
        output[-frame_len:] += audio[-frame_len:] * gain * 0.5

    return output.astype(np.float32)


# ─── VAD ──────────────────────────────────────────────────────────────────────

def _get_speech_regions(audio: np.ndarray, sr: int, aggressiveness: int = 2) -> List[Tuple[int, int]]:
    """
    Use webrtcvad to find speech regions.
    Returns list of (start_sample, end_sample) tuples.
    Falls back to whole-file if VAD unavailable.
    """
    try:
        import webrtcvad
        import struct

        vad = webrtcvad.Vad(aggressiveness)
        frame_ms = 20
        frame_samples = int(sr * frame_ms / 1000)

        # webrtcvad requires 16kHz 16-bit mono
        if sr != 16000:
            return [(0, len(audio))]  # fallback — preprocessing should set sr=16k

        pcm = (audio * 32767).astype(np.int16)
        regions = []
        in_speech = False
        start = 0

        for i in range(0, len(pcm) - frame_samples, frame_samples):
            frame = pcm[i : i + frame_samples]
            frame_bytes = struct.pack(f"{len(frame)}h", *frame)
            is_speech = vad.is_speech(frame_bytes, sr)

            if is_speech and not in_speech:
                start = i
                in_speech = True
            elif not is_speech and in_speech:
                # Pad 200ms around speech
                pad = int(sr * 0.2)
                regions.append((max(0, start - pad), min(len(audio), i + pad)))
                in_speech = False

        if in_speech:
            regions.append((start, len(audio)))

        return regions if regions else [(0, len(audio))]

    except ImportError:
        return [(0, len(audio))]
    except Exception as e:
        logger.warning(f"VAD failed: {e}")
        return [(0, len(audio))]


# ─── Source separation ────────────────────────────────────────────────────────

def _separate_sources(audio: np.ndarray, sr: int, model_name: str = "demucs") -> np.ndarray:
    """
    Use Demucs to separate the 'vocals' stem from the mixture.
    Falls back to input audio if not available.
    """
    if model_name == "none":
        return audio

    try:
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        logger.info("Running Demucs source separation...")
        model = get_model("htdemucs")
        model.eval()

        # Demucs expects (batch, channels, samples) at 44100 Hz
        # We'll resample up temporarily
        import librosa
        audio_44k = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        tensor = torch.from_numpy(audio_44k).float().unsqueeze(0).unsqueeze(0)
        tensor = tensor.expand(-1, 2, -1)  # mono → stereo

        with torch.no_grad():
            sources = apply_model(model, tensor, device="cpu", shifts=1)

        # sources shape: (batch, stems, channels, samples)
        # stems: drums, bass, other, vocals
        vocals_idx = model.sources.index("vocals")
        vocals = sources[0, vocals_idx].mean(0).numpy()

        # Resample back
        vocals_16k = librosa.resample(vocals, orig_sr=44100, target_sr=sr)
        # Trim/pad to match original length
        if len(vocals_16k) > len(audio):
            vocals_16k = vocals_16k[: len(audio)]
        elif len(vocals_16k) < len(audio):
            vocals_16k = np.pad(vocals_16k, (0, len(audio) - len(vocals_16k)))

        logger.info("Source separation complete")
        return vocals_16k.astype(np.float32)

    except ImportError:
        logger.warning("demucs not installed — skipping source separation")
        return audio
    except Exception as e:
        logger.error(f"Source separation failed: {e}")
        return audio


# ─── Transcription ────────────────────────────────────────────────────────────

def _transcribe(audio_path: Path, model_size: str = "base", language: Optional[str] = None) -> List[TranscriptSegment]:
    """Run Whisper transcription and return timestamped segments."""
    try:
        import whisper

        logger.info(f"Loading Whisper {model_size}...")
        model = whisper.load_model(model_size)
        logger.info("Transcribing...")

        result = model.transcribe(
            str(audio_path),
            language=language,
            verbose=False,
            word_timestamps=False,
            condition_on_previous_text=True,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
        )

        segments = []
        for seg in result.get("segments", []):
            segments.append(
                TranscriptSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                    confidence=1 - seg.get("no_speech_prob", 0),
                )
            )
        return segments

    except ImportError:
        logger.warning("openai-whisper not installed — skipping transcription")
        return []
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return []


# ─── Main pipeline ────────────────────────────────────────────────────────────

class AudioPipeline:
    def __init__(self, model_loader: ModelLoader, job_manager: JobManager):
        self.model_loader = model_loader
        self.job_manager = job_manager
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def run(self, job_id: str, input_path: Path, config: PipelineConfig):
        """Run the full enhancement pipeline asynchronously."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                self._executor,
                self._run_sync,
                job_id, input_path, config,
            )
        except Exception as e:
            logger.exception(f"Pipeline failed for job {job_id}")
            self.job_manager.fail_job(job_id, str(e))

    def _run_sync(self, job_id: str, input_path: Path, config: PipelineConfig):
        """Synchronous pipeline (runs in thread pool)."""
        t0 = time.time()
        jm = self.job_manager

        try:
            # ── Stage 1: Load & preprocess ────────────────────────────────────
            jm.set_progress(job_id, 5, "preprocessing", "Loading audio file...")
            audio, sr = _load_audio(input_path, config.target_sample_rate, config.mono)

            if config.normalize_input:
                audio = _normalize(audio, target_db=-6.0)

            total_duration = len(audio) / sr
            jm.update_job(job_id, duration_seconds=total_duration)
            logger.info(f"[{job_id}] Audio loaded: {total_duration:.1f}s")

            # ── Stage 2: Chunking ─────────────────────────────────────────────
            jm.set_progress(job_id, 10, "chunking", "Splitting into chunks...")
            chunks = _split_chunks(audio, sr, config.chunk_duration_seconds, config.overlap_seconds)
            n_chunks = len(chunks)
            logger.info(f"[{job_id}] Split into {n_chunks} chunks")

            # ── Stage 3–7: Process each chunk ─────────────────────────────────
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_progress = 10 + (i / n_chunks) * 70
                jm.set_progress(
                    job_id, chunk_progress, "enhancing",
                    f"Processing chunk {i+1}/{n_chunks}..."
                )
                processed = self._process_chunk(chunk, sr, config, chunk_idx=i)
                processed_chunks.append(processed)

            # ── Stage 8: Reassemble ───────────────────────────────────────────
            jm.set_progress(job_id, 82, "assembling", "Reassembling chunks...")
            enhanced = _crossfade_join(processed_chunks, config.overlap_seconds, sr)

            # Final normalization
            if config.output_normalize:
                enhanced = _normalize(enhanced, target_db=config.output_normalize_db)

            # ── Stage 9: Save output ──────────────────────────────────────────
            jm.set_progress(job_id, 88, "saving", "Saving enhanced audio...")
            output_dir = file_handler.get_output_dir(job_id)
            output_dir.mkdir(parents=True, exist_ok=True)

            audio_out = output_dir / "enhanced.wav"
            _save_wav(audio_out, enhanced, sr)
            output_files = [str(audio_out)]

            # ── Stage 10: Transcription ────────────────────────────────────────
            if config.transcribe:
                jm.set_progress(job_id, 90, "transcribing", "Transcribing speech...")
                segments = _transcribe(audio_out, config.whisper_model, config.language)

                # Save JSON transcript
                transcript_data = {
                    "job_id": job_id,
                    "duration": total_duration,
                    "segments": [s.dict() for s in segments],
                    "full_text": " ".join(s.text for s in segments),
                }
                json_out = output_dir / "transcript.json"
                json_out.write_text(json.dumps(transcript_data, indent=2))
                output_files.append(str(json_out))

                # Save plain text
                txt_out = output_dir / "transcript.txt"
                lines = [f"[{s.start:.1f}s – {s.end:.1f}s] {s.text}" for s in segments]
                txt_out.write_text("\n".join(lines))
                output_files.append(str(txt_out))

            elapsed = time.time() - t0
            jm.complete_job(job_id, output_files=output_files, duration=total_duration)
            logger.info(f"[{job_id}] Pipeline complete in {elapsed:.1f}s")

        except Exception as e:
            logger.exception(f"[{job_id}] Pipeline error")
            jm.fail_job(job_id, str(e))
            raise

    def _process_chunk(
        self, chunk: np.ndarray, sr: int, config: PipelineConfig, chunk_idx: int
    ) -> np.ndarray:
        """Apply all enhancement stages to a single chunk."""

        # Skip essentially silent chunks
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms < 1e-6:
            return chunk

        audio = chunk.copy()

        # Source separation (only first chunk loads model — cached)
        if config.source_separation and chunk_idx == 0:
            # Only do full separation on first chunk to detect if useful
            separated = _separate_sources(audio, sr, config.separation_model)
            # Blend: if separated is much quieter, use a mix
            sep_rms = np.sqrt(np.mean(separated ** 2))
            orig_rms = np.sqrt(np.mean(audio ** 2))
            if sep_rms > orig_rms * 0.05:
                audio = separated * 0.7 + audio * 0.3
        elif config.source_separation and chunk_idx > 0:
            # For subsequent chunks, just apply lighter version
            pass

        # Noise reduction
        if config.noise_reduction:
            audio = _noise_reduce(audio, sr, config.noise_reduction_strength, config.preserve_weak_signals)

        # Spectral subtraction
        if config.spectral_subtraction:
            audio = _spectral_subtraction(audio, sr, alpha=1.5)

        # Wiener filter
        if config.wiener_filter:
            audio = _wiener_filter(audio, sr)

        # Speech frequency boost
        if config.speech_freq_boost:
            audio = _boost_speech_frequencies(audio, sr, config.voice_boost_db)

        # Dynamic gain for quiet signals
        if config.dynamic_gain:
            audio = _apply_dynamic_gain(audio, sr, ratio=config.dynamic_gain_ratio)

        # Prevent clipping after all gains
        peak = np.max(np.abs(audio))
        if peak > 0.98:
            audio = audio * (0.95 / peak)

        return audio.astype(np.float32)
