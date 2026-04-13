"""
ModelLoader — lazy-loads ML models and caches them in memory.

Models:
  - whisper (speech recognition)
  - noisereduce (no download needed, CPU only)
  - demucs (source separation — optional, large download)
"""

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger("audio_enhancer.models")


class ModelLoader:
    """Thread-safe lazy model registry."""

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._lock = threading.Lock()

    @property
    def loaded_models(self) -> list:
        with self._lock:
            return list(self._models.keys())

    def preload(self):
        """
        Attempt to preload lightweight models at startup.
        Heavy models (demucs, whisper-medium+) load on demand.
        """
        # Verify noisereduce available
        try:
            import noisereduce  # noqa
            logger.info("✅ noisereduce available")
        except ImportError:
            logger.warning("⚠️  noisereduce not installed")

        # Verify librosa
        try:
            import librosa  # noqa
            logger.info("✅ librosa available")
        except ImportError:
            logger.warning("⚠️  librosa not installed")

        # Verify soundfile
        try:
            import soundfile  # noqa
            logger.info("✅ soundfile available")
        except ImportError:
            logger.warning("⚠️  soundfile not installed")

        # Verify scipy
        try:
            import scipy  # noqa
            logger.info("✅ scipy available")
        except ImportError:
            logger.warning("⚠️  scipy not installed")

    def get(self, name: str) -> Optional[Any]:
        with self._lock:
            return self._models.get(name)

    def set(self, name: str, model: Any):
        with self._lock:
            self._models[name] = model
            logger.info(f"Cached model: {name}")

    def load_whisper(self, model_size: str = "base"):
        """Load and cache a Whisper model."""
        key = f"whisper_{model_size}"
        with self._lock:
            if key in self._models:
                return self._models[key]

        try:
            import whisper
            logger.info(f"Loading Whisper {model_size}...")
            model = whisper.load_model(model_size)
            self.set(key, model)
            return model
        except ImportError:
            raise RuntimeError("openai-whisper not installed. Run: pip install openai-whisper")

    def load_demucs(self, model_name: str = "htdemucs"):
        """Load and cache a Demucs model."""
        key = f"demucs_{model_name}"
        with self._lock:
            if key in self._models:
                return self._models[key]

        try:
            from demucs.pretrained import get_model
            logger.info(f"Loading Demucs {model_name}...")
            model = get_model(model_name)
            model.eval()
            self.set(key, model)
            return model
        except ImportError:
            raise RuntimeError("demucs not installed. Run: pip install demucs")
