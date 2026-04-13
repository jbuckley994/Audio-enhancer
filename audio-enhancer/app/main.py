"""
AudioEnhancer - Production-Grade AI Audio Processing System
Main FastAPI application server
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.job_manager import JobManager
from app.core.pipeline import AudioPipeline
from app.core.command_parser import CommandParser
from app.models.schemas import (
    JobStatus, ProcessingCommand, JobResponse,
    TranscriptResponse, PipelineConfig
)
from app.utils.file_handler import FileHandler
from app.utils.model_loader import ModelLoader

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("audio_enhancer")

# ─── Global state ──────────────────────────────────────────────────────────────
job_manager = JobManager()
file_handler = FileHandler()
model_loader = ModelLoader()
command_parser: Optional[CommandParser] = None
audio_pipeline: Optional[AudioPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global command_parser, audio_pipeline

    logger.info("🚀 AudioEnhancer starting up...")

    # Ensure directories exist
    file_handler.ensure_dirs()

    # Warm up models (non-blocking — loads lazily on first use if GPU not available)
    try:
        logger.info("Loading models...")
        await asyncio.get_event_loop().run_in_executor(None, model_loader.preload)
        logger.info("✅ Models loaded")
    except Exception as e:
        logger.warning(f"Model preload skipped: {e} — will load on demand")

    command_parser = CommandParser(model_loader)
    audio_pipeline = AudioPipeline(model_loader, job_manager)

    logger.info("✅ AudioEnhancer ready")
    yield

    # Cleanup
    logger.info("Shutting down...")
    job_manager.cleanup_all()


# ─── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AudioEnhancer API",
    description="AI-powered audio enhancement: recover distant/faint voices, remove noise, transcribe",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI
ui_path = Path(__file__).parent.parent / "ui"
if ui_path.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_path), html=True), name="ui")


# ─── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": model_loader.loaded_models,
        "active_jobs": job_manager.active_count(),
    }


# ─── Upload ────────────────────────────────────────────────────────────────────
@app.post("/upload", response_model=JobResponse)
async def upload_audio(
    file: UploadFile = File(...),
    command: str = Form(default="enhance distant voices and reduce background noise"),
):
    """
    Upload an audio file and specify a processing command.

    Supports: .wav, .mp3, .m4a, .flac, .ogg, .aac
    Max size: configurable (default 2 GB)
    Duration: no hard limit — chunked processing handles hours-long files
    """
    # Validate file type
    allowed_ext = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4", ".webm"}
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in allowed_ext:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {allowed_ext}")

    # Create job
    job_id = str(uuid.uuid4())
    job_manager.create_job(job_id, file.filename or "audio")

    # Save upload
    try:
        input_path = await file_handler.save_upload(file, job_id)
    except Exception as e:
        job_manager.fail_job(job_id, str(e))
        raise HTTPException(500, f"Failed to save file: {e}")

    # Parse command into pipeline config
    try:
        pipeline_config = await command_parser.parse(command)
    except Exception as e:
        logger.warning(f"Command parse failed, using defaults: {e}")
        pipeline_config = PipelineConfig()

    job_manager.update_job(job_id, status="queued", command=command, config=pipeline_config.dict())

    return JobResponse(
        job_id=job_id,
        status="queued",
        message=f"File uploaded. Processing will begin shortly.",
        command_interpreted=pipeline_config.description,
    )


# ─── Process ───────────────────────────────────────────────────────────────────
@app.post("/process/{job_id}")
async def process_audio(
    job_id: str,
    background_tasks: BackgroundTasks,
):
    """Start background processing for an uploaded job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] not in ("queued", "failed"):
        raise HTTPException(400, f"Job is already {job['status']}")

    config = PipelineConfig(**job.get("config", {}))
    input_path = file_handler.get_input_path(job_id)

    background_tasks.add_task(
        audio_pipeline.run,
        job_id=job_id,
        input_path=input_path,
        config=config,
    )

    job_manager.update_job(job_id, status="processing", progress=0)
    return {"job_id": job_id, "status": "processing", "message": "Processing started"}


@app.post("/upload-and-process", response_model=JobResponse)
async def upload_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    command: str = Form(default="enhance distant voices and reduce background noise"),
    transcribe: bool = Form(default=True),
):
    """Convenience: upload + immediately queue processing."""
    resp = await upload_audio(file=file, command=command)
    job_id = resp.job_id

    # Inject transcribe flag into config
    job = job_manager.get_job(job_id)
    cfg = job.get("config", {})
    cfg["transcribe"] = transcribe
    job_manager.update_job(job_id, config=cfg)

    config = PipelineConfig(**cfg)
    input_path = file_handler.get_input_path(job_id)

    background_tasks.add_task(
        audio_pipeline.run,
        job_id=job_id,
        input_path=input_path,
        config=config,
    )
    job_manager.update_job(job_id, status="processing", progress=0)

    return JobResponse(
        job_id=job_id,
        status="processing",
        message="Upload complete. Processing started in background.",
        command_interpreted=resp.command_interpreted,
    )


# ─── Status ────────────────────────────────────────────────────────────────────
@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Poll processing status and progress."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JobStatus(**job)


@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    return job_manager.list_jobs()


# ─── Download ──────────────────────────────────────────────────────────────────
@app.get("/download/{job_id}/audio")
async def download_audio(job_id: str):
    """Download the enhanced audio file."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "completed":
        raise HTTPException(400, f"Job not completed (status: {job['status']})")

    output_path = file_handler.get_output_path(job_id, "enhanced.wav")
    if not output_path.exists():
        raise HTTPException(404, "Output file not found")

    return FileResponse(
        path=str(output_path),
        media_type="audio/wav",
        filename=f"enhanced_{job_id[:8]}.wav",
    )


@app.get("/download/{job_id}/transcript")
async def download_transcript(job_id: str, format: str = "json"):
    """Download the transcript (json or txt)."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "completed":
        raise HTTPException(400, f"Job not completed (status: {job['status']})")

    ext = "txt" if format == "txt" else "json"
    output_path = file_handler.get_output_path(job_id, f"transcript.{ext}")
    if not output_path.exists():
        raise HTTPException(404, "Transcript not found. Was transcription enabled?")

    media_type = "text/plain" if ext == "txt" else "application/json"
    return FileResponse(
        path=str(output_path),
        media_type=media_type,
        filename=f"transcript_{job_id[:8]}.{ext}",
    )


# ─── Command parsing (standalone) ─────────────────────────────────────────────
@app.post("/parse-command")
async def parse_command_endpoint(body: ProcessingCommand):
    """Preview how a command will be interpreted."""
    config = await command_parser.parse(body.command)
    return {"command": body.command, "config": config.dict()}


# ─── Delete job ────────────────────────────────────────────────────────────────
@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    file_handler.cleanup_job(job_id)
    job_manager.remove_job(job_id)
    return {"deleted": job_id}
