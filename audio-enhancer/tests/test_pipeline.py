#!/usr/bin/env python3
"""
Unit tests + integration smoke test for AudioEnhancer.
Run with: python tests/test_pipeline.py
"""

import sys
import os
import unittest
import numpy as np
import tempfile
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def make_test_audio(sr=16000, duration=3.0, add_noise=True) -> np.ndarray:
    """Generate a synthetic test signal: 440 Hz tone + pink noise."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Faint voice signal (440 Hz + harmonics at low amplitude)
    voice = 0.05 * np.sin(2 * np.pi * 440 * t)
    voice += 0.02 * np.sin(2 * np.pi * 880 * t)
    voice += 0.01 * np.sin(2 * np.pi * 1320 * t)
    if add_noise:
        # Loud noise (simulates foreground masker)
        noise = 0.4 * np.random.randn(len(t))
        return (voice + noise).astype(np.float32)
    return voice.astype(np.float32)


class TestNormalize(unittest.TestCase):
    def test_peak_normalization(self):
        from app.core.pipeline import _normalize
        audio = np.array([0.1, -0.5, 0.3, -0.8], dtype=np.float32)
        result = _normalize(audio, target_db=-3.0)
        peak = np.max(np.abs(result))
        expected = 10 ** (-3.0 / 20.0)
        self.assertAlmostEqual(peak, expected, places=4)

    def test_silent_audio_unchanged(self):
        from app.core.pipeline import _normalize
        audio = np.zeros(1000, dtype=np.float32)
        result = _normalize(audio)
        np.testing.assert_array_equal(result, audio)


class TestChunking(unittest.TestCase):
    def test_chunk_count(self):
        from app.core.pipeline import _split_chunks
        audio = np.zeros(16000 * 130, dtype=np.float32)  # 130s
        chunks = _split_chunks(audio, sr=16000, chunk_sec=60, overlap_sec=2)
        self.assertGreater(len(chunks), 1)

    def test_chunk_overlap(self):
        from app.core.pipeline import _split_chunks
        audio = np.zeros(16000 * 65, dtype=np.float32)  # 65s
        chunks = _split_chunks(audio, sr=16000, chunk_sec=60, overlap_sec=2)
        # First chunk: 60s, second chunk: remaining + overlap
        self.assertEqual(len(chunks[0]), 16000 * 60)

    def test_short_audio_single_chunk(self):
        from app.core.pipeline import _split_chunks
        audio = np.zeros(16000 * 30, dtype=np.float32)
        chunks = _split_chunks(audio, sr=16000, chunk_sec=60, overlap_sec=2)
        self.assertEqual(len(chunks), 1)


class TestCrossfadeJoin(unittest.TestCase):
    def test_single_chunk(self):
        from app.core.pipeline import _crossfade_join
        chunk = np.ones(16000, dtype=np.float32)
        result = _crossfade_join([chunk], overlap_sec=2, sr=16000)
        np.testing.assert_array_equal(result, chunk)

    def test_two_chunks_length(self):
        from app.core.pipeline import _crossfade_join
        c1 = np.ones(32000, dtype=np.float32)
        c2 = np.ones(32000, dtype=np.float32)
        result = _crossfade_join([c1, c2], overlap_sec=2, sr=16000)
        # Total = 32000 + 32000 - 32000(overlap) + 32000(overlap-region) roughly
        self.assertGreater(len(result), 32000)

    def test_empty_list(self):
        from app.core.pipeline import _crossfade_join
        result = _crossfade_join([], overlap_sec=2, sr=16000)
        self.assertEqual(len(result), 0)


class TestSpectralSubtraction(unittest.TestCase):
    def test_output_shape(self):
        from app.core.pipeline import _spectral_subtraction
        try:
            import librosa  # noqa
        except ImportError:
            self.skipTest("librosa not installed")
        sr = 16000
        audio = make_test_audio(sr, 2.0)
        result = _spectral_subtraction(audio, sr)
        self.assertEqual(len(result), len(audio))

    def test_output_dtype(self):
        from app.core.pipeline import _spectral_subtraction
        try:
            import librosa  # noqa
        except ImportError:
            self.skipTest("librosa not installed")
        audio = make_test_audio(16000, 1.0)
        result = _spectral_subtraction(audio, 16000)
        self.assertEqual(result.dtype, np.float32)


class TestWienerFilter(unittest.TestCase):
    def test_snr_improves(self):
        from app.core.pipeline import _wiener_filter
        try:
            import librosa  # noqa
        except ImportError:
            self.skipTest("librosa not installed")
        sr = 16000
        # Pure voice signal
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        voice = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        noise = 0.5 * np.random.randn(len(t)).astype(np.float32)
        noisy = voice + noise

        filtered = _wiener_filter(noisy, sr)
        # Filtered should have lower energy (noise suppressed)
        self.assertLess(np.mean(filtered**2), np.mean(noisy**2))


class TestSpeechFreqBoost(unittest.TestCase):
    def test_output_not_clipping(self):
        from app.core.pipeline import _boost_speech_frequencies
        try:
            from scipy.signal import butter  # noqa
        except ImportError:
            self.skipTest("scipy not installed")
        audio = make_test_audio(16000, 2.0, add_noise=False)
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.5  # normalize to 50%
        result = _boost_speech_frequencies(audio, 16000, boost_db=6.0)
        self.assertLessEqual(np.max(np.abs(result)), 1.01)  # no hard clipping


class TestNoiseReduce(unittest.TestCase):
    def test_reduces_rms(self):
        from app.core.pipeline import _noise_reduce
        try:
            import noisereduce  # noqa
        except ImportError:
            self.skipTest("noisereduce not installed")
        sr = 16000
        audio = make_test_audio(sr, 3.0, add_noise=True)
        reduced = _noise_reduce(audio, sr, strength=0.75, preserve_weak=True)
        self.assertLess(np.sqrt(np.mean(reduced**2)), np.sqrt(np.mean(audio**2)))

    def test_output_shape_preserved(self):
        from app.core.pipeline import _noise_reduce
        try:
            import noisereduce  # noqa
        except ImportError:
            self.skipTest("noisereduce not installed")
        sr = 16000
        audio = make_test_audio(sr, 3.0)
        reduced = _noise_reduce(audio, sr, strength=0.5, preserve_weak=True)
        self.assertEqual(len(reduced), len(audio))


class TestDynamicGain(unittest.TestCase):
    def test_quiet_signal_amplified(self):
        from app.core.pipeline import _apply_dynamic_gain
        sr = 16000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        # Very quiet voice signal
        quiet = (0.001 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        amplified = _apply_dynamic_gain(quiet, sr, ratio=3.0, threshold_db=-40.0)
        # Output should have more energy than input
        self.assertGreater(np.mean(amplified**2), np.mean(quiet**2))


class TestSaveLoad(unittest.TestCase):
    def test_roundtrip_wav(self):
        from app.core.pipeline import _save_wav, _load_audio
        try:
            import soundfile  # noqa
            import librosa  # noqa
        except ImportError:
            self.skipTest("soundfile/librosa not installed")

        sr = 16000
        audio = make_test_audio(sr, 1.0, add_noise=False)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)
        try:
            _save_wav(path, audio, sr)
            loaded, loaded_sr = _load_audio(path, sr)
            self.assertEqual(loaded_sr, sr)
            self.assertEqual(len(loaded), len(audio))
        finally:
            path.unlink(missing_ok=True)


class TestJobManager(unittest.TestCase):
    def setUp(self):
        from app.core.job_manager import JobManager
        self.jm = JobManager()

    def test_create_and_get(self):
        self.jm.create_job("j1", "test.wav")
        job = self.jm.get_job("j1")
        self.assertIsNotNone(job)
        self.assertEqual(job["job_id"], "j1")
        self.assertEqual(job["status"], "created")

    def test_update(self):
        self.jm.create_job("j2", "test.wav")
        self.jm.update_job("j2", status="processing", progress=50.0)
        job = self.jm.get_job("j2")
        self.assertEqual(job["status"], "processing")
        self.assertEqual(job["progress"], 50.0)

    def test_complete(self):
        self.jm.create_job("j3", "test.wav")
        self.jm.complete_job("j3", output_files=["a.wav"], duration=120.0)
        job = self.jm.get_job("j3")
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["progress"], 100.0)

    def test_fail(self):
        self.jm.create_job("j4", "test.wav")
        self.jm.fail_job("j4", "Something went wrong")
        job = self.jm.get_job("j4")
        self.assertEqual(job["status"], "failed")
        self.assertIn("Something", job["error"])

    def test_not_found(self):
        result = self.jm.get_job("nonexistent")
        self.assertIsNone(result)

    def test_active_count(self):
        self.jm.create_job("j5", "a.wav")
        self.jm.create_job("j6", "b.wav")
        self.jm.update_job("j5", status="processing")
        self.jm.update_job("j6", status="processing")
        self.assertEqual(self.jm.active_count(), 2)


class TestCommandParserRules(unittest.TestCase):
    """Test rule-based command parsing (no API key needed)."""

    def _parse(self, command):
        """Helper: use rule-based parsing directly."""
        from app.core.command_parser import CommandParser
        from app.utils.model_loader import ModelLoader
        parser = CommandParser(ModelLoader())
        # Force rule-based by clearing API keys
        parser._anthropic_key = None
        parser._openai_key = None
        import asyncio
        return asyncio.get_event_loop().run_until_complete(parser.parse(command))

    def test_distant_voice_command(self):
        cfg = self._parse("enhance distant voices")
        self.assertTrue(cfg.voice_boost)

    def test_noise_reduction_command(self):
        cfg = self._parse("remove background noise")
        self.assertTrue(cfg.noise_reduction)

    def test_transcribe_command(self):
        cfg = self._parse("transcribe the audio")
        self.assertTrue(cfg.transcribe)

    def test_no_transcribe(self):
        cfg = self._parse("audio only, no transcription")
        self.assertFalse(cfg.transcribe)

    def test_heavy_noise(self):
        cfg = self._parse("heavy noise reduction")
        self.assertGreaterEqual(cfg.noise_reduction_strength, 0.85)

    def test_gentle(self):
        cfg = self._parse("gentle enhancement, preserve everything")
        self.assertTrue(cfg.preserve_weak_signals)


class TestSchemas(unittest.TestCase):
    def test_pipeline_config_defaults(self):
        from app.models.schemas import PipelineConfig
        cfg = PipelineConfig()
        self.assertEqual(cfg.target_sample_rate, 16000)
        self.assertTrue(cfg.noise_reduction)
        self.assertTrue(cfg.voice_boost)
        self.assertEqual(cfg.whisper_model, "base")

    def test_pipeline_config_override(self):
        from app.models.schemas import PipelineConfig
        cfg = PipelineConfig(noise_reduction_strength=0.9, voice_boost_db=12.0)
        self.assertEqual(cfg.noise_reduction_strength, 0.9)
        self.assertEqual(cfg.voice_boost_db, 12.0)


class TestIntegrationPipeline(unittest.TestCase):
    """Full end-to-end pipeline smoke test (no ML models needed)."""

    def test_full_pipeline_numpy(self):
        """Process a synthetic noisy signal through all DSP stages."""
        required = []
        for mod in ("librosa", "soundfile", "scipy"):
            try:
                __import__(mod)
            except ImportError:
                required.append(mod)
        if required:
            self.skipTest(f"Missing: {required}")

        from app.core.pipeline import (
            _normalize, _split_chunks, _crossfade_join,
            _spectral_subtraction, _wiener_filter,
            _boost_speech_frequencies, _apply_dynamic_gain,
            _save_wav, _load_audio,
        )

        sr = 16000
        audio = make_test_audio(sr, duration=5.0, add_noise=True)
        audio = _normalize(audio, -6.0)

        chunks = _split_chunks(audio, sr, chunk_sec=2, overlap_sec=0)
        processed = []
        for chunk in chunks:
            c = _spectral_subtraction(chunk, sr)
            c = _wiener_filter(c, sr)
            c = _boost_speech_frequencies(c, sr, boost_db=6.0)
            c = _apply_dynamic_gain(c, sr)
            peak = np.max(np.abs(c))
            if peak > 0.98:
                c = c * (0.95 / peak)
            processed.append(c)

        result = _crossfade_join(processed, overlap_sec=0, sr=sr)
        result = _normalize(result, -3.0)

        self.assertEqual(len(result), len(audio))
        self.assertLessEqual(np.max(np.abs(result)), 1.01)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)
        try:
            _save_wav(path, result, sr)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    print("=" * 60)
    print("AudioEnhancer Test Suite")
    print("=" * 60)
    unittest.main(verbosity=2)
