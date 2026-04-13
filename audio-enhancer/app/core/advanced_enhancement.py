"""
advanced_enhancement.py — Extra signal processing for extreme low-SNR scenarios.

These techniques are applied on top of the main pipeline when dealing with:
  - Voices recorded through walls/doors (severe high-frequency attenuation)
  - Extremely low SNR (< -10 dB)
  - Intermittent / partially masked speech

Techniques:
  1. Multi-band dynamic processing
  2. Harmonic enhancement (comb filter aligned to f0)
  3. Phase-sensitive Wiener filter
  4. Log-MMSE spectral estimator
  5. Bandwidth extension (recover attenuated high frequencies)
"""

import logging
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger("audio_enhancer.advanced")


# ─── Multi-band compander ─────────────────────────────────────────────────────

def multiband_enhance(audio: np.ndarray, sr: int, n_bands: int = 8) -> np.ndarray:
    """
    Split audio into N frequency bands and apply independent upward compression
    to each band. This recovers voice in bands where it's buried.

    Based on: Sub-band processing for speech enhancement.
    """
    try:
        from scipy.signal import butter, sosfilt

        band_outputs = []
        freqs = np.logspace(np.log10(100), np.log10(min(8000, sr / 2 - 100)), n_bands + 1)

        for i in range(n_bands):
            lo = freqs[i] / (sr / 2)
            hi = freqs[i + 1] / (sr / 2)
            if lo >= 0.99 or hi >= 0.99 or lo >= hi:
                continue

            # Extract band
            sos = butter(4, [lo, hi], btype="bandpass", output="sos")
            band = sosfilt(sos, audio)

            # Compute band energy over short frames
            frame_len = int(sr * 0.025)
            rms_vals = []
            for j in range(0, len(band) - frame_len, frame_len // 2):
                rms_vals.append(np.sqrt(np.mean(band[j : j + frame_len] ** 2)))

            if not rms_vals:
                band_outputs.append(band)
                continue

            median_rms = np.median(rms_vals)
            # Boost bands where speech energy is low relative to overall
            boost = np.clip(0.02 / (median_rms + 1e-9), 1.0, 6.0)
            band_outputs.append(band * boost)

        if not band_outputs:
            return audio

        combined = np.sum(band_outputs, axis=0)
        # Renormalize to prevent clipping
        peak = np.max(np.abs(combined))
        if peak > 0.95:
            combined = combined * (0.90 / peak)

        return combined.astype(np.float32)

    except ImportError:
        logger.warning("scipy not installed — skipping multiband enhance")
        return audio
    except Exception as e:
        logger.error(f"Multiband enhance failed: {e}")
        return audio


# ─── Pitch-synchronized harmonic enhancement ──────────────────────────────────

def harmonic_enhancement(
    audio: np.ndarray, sr: int, f0_min: float = 80.0, f0_max: float = 400.0
) -> np.ndarray:
    """
    Estimate fundamental frequency (f0) and boost harmonics.

    Useful for: voices recorded through walls (high-frequency attenuation means
    harmonics are suppressed; boosting them restores perceived intelligibility).

    Uses autocorrelation-based pitch detection.
    """
    try:
        import librosa

        frame_len = int(sr * 0.025)  # 25ms
        hop = frame_len // 2
        output = audio.copy()

        # Extract f0 using pyin (probabilistic YIN)
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=f0_min,
                fmax=f0_max,
                sr=sr,
                frame_length=frame_len,
                hop_length=hop,
            )
        except Exception:
            return audio

        # STFT
        n_fft = 2048
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
        magnitude, phase = np.abs(stft), np.angle(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        enhanced_mag = magnitude.copy()

        # For each voiced frame, boost harmonics
        for t_idx in range(min(stft.shape[1], len(f0))):
            if not voiced_flag[t_idx] or np.isnan(f0[t_idx]):
                continue
            f_fund = f0[t_idx]
            # Boost up to 8th harmonic
            for harmonic in range(1, 9):
                f_harm = f_fund * harmonic
                if f_harm > sr / 2:
                    break
                # Find closest FFT bin
                bin_idx = np.argmin(np.abs(freqs - f_harm))
                # Boost ±2 bins around harmonic
                lo = max(0, bin_idx - 2)
                hi = min(magnitude.shape[0], bin_idx + 3)
                boost = 1.0 + (0.8 / harmonic)  # diminishing boost for higher harmonics
                enhanced_mag[lo:hi, t_idx] *= boost

        # Reconstruct
        enhanced_stft = enhanced_mag * np.exp(1j * phase)
        enhanced = librosa.istft(enhanced_stft, hop_length=hop, length=len(audio))

        # Prevent clipping
        peak = np.max(np.abs(enhanced))
        if peak > 0.97:
            enhanced = enhanced * (0.95 / peak)

        return enhanced.astype(np.float32)

    except ImportError:
        logger.warning("librosa not installed — skipping harmonic enhancement")
        return audio
    except Exception as e:
        logger.error(f"Harmonic enhancement failed: {e}")
        return audio


# ─── Log-MMSE speech estimator ────────────────────────────────────────────────

def log_mmse_enhance(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Log-MMSE (Minimum Mean Square Error) speech enhancement.

    Better than basic spectral subtraction at preserving naturalness:
    - Estimates a priori and a posteriori SNR per frequency bin
    - Applies optimal Wiener gain in log domain
    - Reduces musical noise artifacts

    Reference: Ephraim & Malah (1985).
    """
    try:
        import librosa

        n_fft = 512
        hop = 128
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
        power = np.abs(stft) ** 2
        n_frames = power.shape[1]

        # Initial noise estimate: first few frames assumed to be noise
        noise_frames = max(6, n_frames // 20)
        noise_power = np.mean(power[:, :noise_frames], axis=1, keepdims=True)
        noise_power = np.maximum(noise_power, 1e-10)

        # Smoothing factor for noise estimate update
        alpha_n = 0.98
        # Prior SNR smoothing
        alpha_p = 0.98

        prior_snr = np.ones_like(noise_power)
        gain = np.ones_like(power)
        enhanced_power = np.zeros_like(power)

        for t in range(n_frames):
            frame_power = power[:, t : t + 1]

            # A posteriori SNR
            post_snr = frame_power / noise_power

            # A priori SNR (decision-directed)
            prior_snr = alpha_p * (gain[:, -1:] ** 2) * enhanced_power[:, max(0, t - 1) : t] / noise_power + \
                        (1 - alpha_p) * np.maximum(post_snr - 1, 0)
            prior_snr = np.maximum(prior_snr, 1e-4)

            # MMSE-LSA gain
            nu = prior_snr / (1 + prior_snr) * post_snr
            # Gain approximation: G = prior_snr/(1+prior_snr) * exp(0.5 * E1(nu))
            # Simplified: G ≈ sqrt(prior_snr / (1 + prior_snr))
            g = np.sqrt(prior_snr / (1 + prior_snr))
            g = np.maximum(g, 0.05)  # noise floor

            gain = g
            enhanced_power[:, t : t + 1] = g ** 2 * frame_power

            # Update noise estimate (only in non-speech frames)
            # Simple energy-based VAD: noise frame if total power is low
            if np.mean(post_snr) < 1.5:
                noise_power = alpha_n * noise_power + (1 - alpha_n) * frame_power

        # Apply gain to magnitude spectrum
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Reshape gain to match stft shape
        if gain.shape[1] == 1:
            gain_full = np.repeat(gain, n_frames, axis=1)
        else:
            gain_full = gain

        enhanced_mag = magnitude * gain_full[:, :n_frames]
        enhanced_stft = enhanced_mag * np.exp(1j * phase)
        enhanced = librosa.istft(enhanced_stft, hop_length=hop, length=len(audio))

        return enhanced.astype(np.float32)

    except Exception as e:
        logger.error(f"Log-MMSE failed: {e}")
        return audio


# ─── Bandwidth extension ─────────────────────────────────────────────────────

def bandwidth_extension(audio: np.ndarray, sr: int, cutoff_hz: float = 3500.0) -> np.ndarray:
    """
    Estimate and reconstruct high-frequency content that was attenuated
    (e.g., by walls/doors acting as low-pass filters).

    Method: Copy sub-band energy pattern from 1-3.5kHz and shift it up
    to 3.5-8kHz with appropriate spectral shaping.

    This is a lightweight approximation of learned bandwidth extension.
    """
    try:
        import librosa
        from scipy.signal import butter, sosfilt

        # Check if high frequencies are truly attenuated
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        mag = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        low_band_mask = (freqs >= 500) & (freqs < cutoff_hz)
        high_band_mask = freqs >= cutoff_hz

        low_energy = np.mean(mag[low_band_mask])
        high_energy = np.mean(mag[high_band_mask])

        # Only apply extension if high frequencies are significantly attenuated
        if high_energy >= low_energy * 0.3:
            return audio  # Extension not needed

        logger.info(f"Bandwidth extension: high/low energy ratio = {high_energy/low_energy:.3f}")

        # Extract the 1.5-3.5kHz band (upper speech harmonics as source)
        lo = 1500 / (sr / 2)
        hi = min(cutoff_hz / (sr / 2), 0.98)
        sos = butter(6, [lo, hi], btype="bandpass", output="sos")
        source_band = sosfilt(sos, audio)

        # Non-linear processing to create harmonic-like content
        # Full-wave rectification + re-filtering shifts frequency content up
        rectified = np.abs(source_band)

        # Filter to target band (cutoff_hz to cutoff_hz*2)
        ext_lo = min(cutoff_hz / (sr / 2), 0.95)
        ext_hi = min((cutoff_hz * 1.8) / (sr / 2), 0.99)

        if ext_lo >= ext_hi:
            return audio

        sos2 = butter(4, [ext_lo, ext_hi], btype="bandpass", output="sos")
        extension = sosfilt(sos2, rectified)

        # Mix in extension at low level
        mix_ratio = 0.15
        enhanced = audio + extension * mix_ratio

        peak = np.max(np.abs(enhanced))
        if peak > 0.97:
            enhanced = enhanced * (0.95 / peak)

        return enhanced.astype(np.float32)

    except Exception as e:
        logger.error(f"Bandwidth extension failed: {e}")
        return audio


# ─── Convenience: apply all advanced stages ────────────────────────────────────

def apply_advanced_enhancement(
    audio: np.ndarray,
    sr: int,
    use_multiband: bool = True,
    use_harmonics: bool = True,
    use_log_mmse: bool = True,
    use_bwe: bool = True,
) -> np.ndarray:
    """
    Apply all advanced enhancement stages in sequence.

    Recommended for: voices through walls/doors, extreme low SNR recordings.
    Add after the main pipeline stages.
    """
    if use_log_mmse:
        logger.info("  → Log-MMSE spectral estimator")
        audio = log_mmse_enhance(audio, sr)

    if use_multiband:
        logger.info("  → Multi-band dynamic enhancement")
        audio = multiband_enhance(audio, sr)

    if use_harmonics:
        logger.info("  → Harmonic pitch enhancement")
        audio = harmonic_enhancement(audio, sr)

    if use_bwe:
        logger.info("  → Bandwidth extension")
        audio = bandwidth_extension(audio, sr)

    return audio
