"""
Pydantic models / schemas for the AudioEnhancer API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProcessingCommand(BaseModel):
    command: str = Field(..., description="Natural language processing instruction")


class PipelineConfig(BaseModel):
    """Resolved pipeline configuration derived from a user command."""

    description: str = Field(
        default="Enhance distant voices and reduce background noise",
        description="Human-readable summary of what will be done",
    )

    # Preprocessing
    target_sample_rate: int = Field(default=16000)
    mono: bool = Field(default=True)
    normalize_input: bool = Field(default=True)

    # VAD
    vad_enabled: bool = Field(default=True)
    vad_aggressiveness: int = Field(default=2, ge=0, le=3)  # webrtcvad levels

    # Source separation
    source_separation: bool = Field(default=True)
    separation_model: str = Field(default="demucs")  # demucs | spleeter | none

    # Noise reduction
    noise_reduction: bool = Field(default=True)
    noise_reduction_strength: float = Field(default=0.75, ge=0.0, le=1.0)
    preserve_weak_signals: bool = Field(default=True)

    # Voice enhancement
    voice_boost: bool = Field(default=True)
    voice_boost_db: float = Field(default=6.0, ge=0.0, le=24.0)
    speech_freq_boost: bool = Field(default=True)  # boost 300–3400 Hz band
    dynamic_gain: bool = Field(default=True)
    dynamic_gain_ratio: float = Field(default=3.0)  # compander ratio

    # Spectral enhancement
    spectral_subtraction: bool = Field(default=True)
    wiener_filter: bool = Field(default=True)

    # Post-processing
    output_normalize: bool = Field(default=True)
    output_normalize_db: float = Field(default=-3.0)

    # Chunking
    chunk_duration_seconds: int = Field(default=60)
    overlap_seconds: int = Field(default=2)
    parallel_chunks: int = Field(default=2)

    # Transcription
    transcribe: bool = Field(default=True)
    whisper_model: str = Field(default="base")  # tiny|base|small|medium|large
    language: Optional[str] = Field(default=None)  # None = auto-detect


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    command_interpreted: str = ""


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued|processing|completed|failed
    progress: float = 0.0  # 0–100
    stage: str = ""
    message: str = ""
    command: str = ""
    filename: str = ""
    config: Dict[str, Any] = {}
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
    output_files: List[str] = []


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    confidence: Optional[float] = None


class TranscriptResponse(BaseModel):
    job_id: str
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: List[TranscriptSegment] = []
    full_text: str = ""
