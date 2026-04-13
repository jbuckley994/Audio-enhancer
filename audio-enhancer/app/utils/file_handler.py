"""
FileHandler — manages upload/output directory structure and async file I/O.
"""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

logger = logging.getLogger("audio_enhancer.files")

BASE_DIR = Path("./data")
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
MAX_CHUNK_SIZE = 1024 * 1024  # 1 MB read chunks for streaming upload


class FileHandler:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base = base_dir or BASE_DIR
        self.uploads = self.base / "uploads"
        self.outputs = self.base / "outputs"

    def ensure_dirs(self):
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data directories: {self.uploads}, {self.outputs}")

    async def save_upload(self, file: UploadFile, job_id: str) -> Path:
        """Stream upload to disk without loading it all in memory."""
        job_upload_dir = self.uploads / job_id
        job_upload_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
        dest = job_upload_dir / f"input{suffix}"

        logger.info(f"Saving upload for job {job_id} → {dest}")
        total = 0
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(MAX_CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)

        logger.info(f"Saved {total / 1_000_000:.1f} MB → {dest}")
        return dest

    def get_input_path(self, job_id: str) -> Path:
        """Find the input file for a job (any audio extension)."""
        job_dir = self.uploads / job_id
        for ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4", ".webm"):
            p = job_dir / f"input{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"No input file found for job {job_id}")

    def get_output_dir(self, job_id: str) -> Path:
        return self.outputs / job_id

    def get_output_path(self, job_id: str, filename: str) -> Path:
        return self.outputs / job_id / filename

    def cleanup_job(self, job_id: str):
        """Remove all files for a job."""
        for d in (self.uploads / job_id, self.outputs / job_id):
            if d.exists():
                shutil.rmtree(d)
                logger.info(f"Cleaned up {d}")
