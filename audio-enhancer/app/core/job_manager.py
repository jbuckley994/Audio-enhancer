"""
JobManager — in-memory job tracking with thread-safe updates.
For production, swap the dict for Redis or a DB.
"""

import threading
import time
from typing import Any, Dict, List, Optional


class JobManager:
    """Thread-safe in-memory job registry."""

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create_job(self, job_id: str, filename: str) -> Dict:
        job = {
            "job_id": job_id,
            "status": "created",
            "progress": 0.0,
            "stage": "upload",
            "message": "Waiting for file",
            "command": "",
            "filename": filename,
            "config": {},
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
            "duration_seconds": None,
            "output_files": [],
        }
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Dict]:
        with self._lock:
            return dict(self._jobs[job_id]) if job_id in self._jobs else None

    def update_job(self, job_id: str, **kwargs) -> bool:
        with self._lock:
            if job_id not in self._jobs:
                return False
            self._jobs[job_id].update(kwargs)
            self._jobs[job_id]["updated_at"] = time.time()
        return True

    def remove_job(self, job_id: str):
        with self._lock:
            self._jobs.pop(job_id, None)

    def list_jobs(self) -> List[Dict]:
        with self._lock:
            return [dict(j) for j in self._jobs.values()]

    # ── Convenience helpers ───────────────────────────────────────────────────

    def set_progress(self, job_id: str, progress: float, stage: str = "", message: str = ""):
        self.update_job(job_id, progress=min(100.0, progress), stage=stage, message=message)

    def complete_job(self, job_id: str, output_files: List[str], duration: float):
        self.update_job(
            job_id,
            status="completed",
            progress=100.0,
            stage="done",
            message="Processing complete",
            output_files=output_files,
            duration_seconds=duration,
        )

    def fail_job(self, job_id: str, error: str):
        self.update_job(
            job_id,
            status="failed",
            stage="error",
            message=f"Processing failed: {error}",
            error=error,
        )

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j["status"] == "processing")

    def cleanup_all(self):
        with self._lock:
            self._jobs.clear()
