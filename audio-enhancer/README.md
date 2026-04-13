# 🎙 AudioEnhancer

**AI-powered audio processing system for recovering distant, faint, and obstructed voices.**

Built with FastAPI, librosa, noisereduce, Whisper, and optional deep learning (Demucs) for source separation.

---

## What It Does

AudioEnhancer is designed for the hardest audio recovery cases:

- 🚪 **Voices through walls and doors** — dynamic gain + spectral subtraction recovers muffled speech
- 🔇 **Loud foreground noise** — Wiener filtering + spectral gating isolates distant signal
- 📻 **Extremely low SNR recordings** — multi-stage pipeline preserves weak signals while reducing noise
- ⏱ **Hours-long files** — chunk-based processing with crossfade reassembly handles any duration
- 📝 **Speech-to-text** — Whisper transcription with timestamps on the enhanced output

---

## Architecture

```
User Upload (any audio format, any length)
         │
         ▼
┌────────────────────────────────────────────┐
│ 1. PREPROCESSING                           │
│    Load → Mono → 16kHz → Normalize        │
└────────────────────────┬───────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────┐
│ 2. CHUNKING                                │
│    Split into 60s chunks with 2s overlap  │
└────────────────────────┬───────────────────┘
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
    [Chunk 1]      [Chunk 2]     [Chunk N]    ← parallel workers
    ┌──────────────────────────────────────┐
    │ 3. SOURCE SEPARATION (Demucs)        │
    │    Extract vocals stem from mixture  │
    ├──────────────────────────────────────┤
    │ 4. NOISE REDUCTION (noisereduce)     │
    │    Spectral gating with noise est.   │
    ├──────────────────────────────────────┤
    │ 5. SPECTRAL SUBTRACTION              │
    │    Classic SS with over-subtraction  │
    ├──────────────────────────────────────┤
    │ 6. WIENER FILTER                     │
    │    Frequency-domain SNR-based gain   │
    ├──────────────────────────────────────┤
    │ 7. SPEECH FREQ BOOST (300–5000 Hz)   │
    │    EQ + presence boost               │
    ├──────────────────────────────────────┤
    │ 8. DYNAMIC GAIN (upward expansion)   │
    │    Amplify quiet frames selectively  │
    └──────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────┐
│ 9. REASSEMBLY                              │
│    Cosine crossfade overlap join           │
│    Peak normalization → -3 dBFS           │
└────────────────────────┬───────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────┐
│ 10. TRANSCRIPTION (Whisper)                │
│     Speech-to-text on enhanced audio      │
│     Timestamped JSON + plain TXT output   │
└────────────────────────┬───────────────────┘
                         │
                         ▼
              enhanced.wav + transcript.json
```

---

## Quick Start

### Prerequisites

- Python 3.10+ 
- ffmpeg (`brew install ffmpeg` / `apt install ffmpeg`)

### 1. Clone & Install

```bash
git clone <repo>
cd audio-enhancer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note on heavy dependencies:**
> - `torch` is ~2GB; skip if you don't need source separation
> - `openai-whisper` downloads models on first use (~140MB for `base`)
> - `demucs` downloads ~1.5GB model on first use; uncomment in requirements.txt to enable

### 2. Configure

```bash
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY for AI command parsing
```

### 3. Run

```bash
# Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Open the web UI
open http://localhost:8000/ui

# Or browse the interactive API docs
open http://localhost:8000/docs
```

---

## Usage

### Web UI

Open `http://localhost:8000/ui` in your browser:

1. Drop an audio file onto the upload zone
2. Type a natural language command (or pick a preset)
3. Adjust processing options
4. Click **Process Audio**
5. Download the enhanced WAV + transcript

### CLI (no server needed)

```bash
# Basic enhancement
python scripts/process_cli.py recording.wav

# Custom command + options
python scripts/process_cli.py recording.mp3 \
  --command "Recover voices through wall, heavy noise" \
  --noise-strength 0.9 \
  --boost-db 12 \
  --whisper-model small \
  --output recovered.wav

# Fast pass (skip transcription + source separation)
python scripts/process_cli.py recording.wav \
  --no-transcribe \
  --no-source-sep \
  --chunk-seconds 120
```

### REST API

```bash
# Upload + process in one call
curl -X POST http://localhost:8000/upload-and-process \
  -F "file=@recording.wav" \
  -F "command=enhance distant voices, heavy noise reduction" \
  -F "transcribe=true"

# → Returns: {"job_id": "abc123...", "status": "processing"}

# Poll status
curl http://localhost:8000/status/abc123...

# Download results when status == "completed"
curl -O http://localhost:8000/download/abc123.../audio
curl -O http://localhost:8000/download/abc123.../transcript

# Preview command interpretation (no processing)
curl -X POST http://localhost:8000/parse-command \
  -H "Content-Type: application/json" \
  -d '{"command": "recover muffled voices through door, gentle on weak signals"}'
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status + loaded models |
| `/upload` | POST | Upload audio file → job_id |
| `/process/{job_id}` | POST | Start processing a queued job |
| `/upload-and-process` | POST | Combined upload + process |
| `/status/{job_id}` | GET | Poll progress (0–100%) and stage |
| `/jobs` | GET | List all jobs |
| `/download/{job_id}/audio` | GET | Download enhanced WAV |
| `/download/{job_id}/transcript` | GET | Download transcript (JSON or TXT) |
| `/parse-command` | POST | Preview command interpretation |
| `/jobs/{job_id}` | DELETE | Delete job and files |

---

## Configuration

### Pipeline Options (via API or config)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `noise_reduction_strength` | 0.75 | 0–1: higher = more aggressive |
| `preserve_weak_signals` | true | Protect faint voices during denoising |
| `voice_boost_db` | 6.0 | EQ boost for speech frequencies |
| `speech_freq_boost` | true | Boost 300–5000 Hz band |
| `dynamic_gain` | true | Upward expansion for quiet sections |
| `spectral_subtraction` | true | Classic spectral subtraction |
| `wiener_filter` | true | Frequency-domain Wiener filter |
| `source_separation` | true | Demucs vocal isolation (if installed) |
| `transcribe` | true | Whisper speech-to-text |
| `whisper_model` | base | tiny/base/small/medium/large |
| `chunk_duration_seconds` | 60 | Chunk size for long-file processing |
| `overlap_seconds` | 2 | Overlap between chunks for crossfade |

---

## Docker

```bash
# Build and run
docker-compose up -d

# With API key
ANTHROPIC_API_KEY=your_key docker-compose up -d

# View logs
docker-compose logs -f audio-enhancer
```

---

## Performance

| File Duration | Processing Time (CPU, base Whisper) |
|--------------|-------------------------------------|
| 5 minutes | ~2–3 minutes |
| 30 minutes | ~12–18 minutes |
| 2 hours | ~50–80 minutes |
| 4 hours | ~100–160 minutes |

GPU acceleration (CUDA/MPS) significantly speeds up Whisper and Demucs.

To use GPU:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## AI Command Parsing

Set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) in `.env` to enable LLM-based command parsing.

Without an API key, the system falls back to regex rule matching, which covers most common commands.

**Example commands the LLM understands:**

- `"The audio was recorded outside a room. Voices are very faint. Transcribe every word."`
- `"Remove the loud TV in the foreground, recover the background conversation"`
- `"Gentle enhancement only — don't destroy any weak audio, just clean up the noise a bit"`
- `"Maximum recovery mode. Do everything possible to make voices intelligible."`

---

## Optimizing for Distant Voice Recovery

For the hardest cases (voices through walls/doors):

```bash
python scripts/process_cli.py recording.wav \
  --command "recover voices through wall, maximum enhancement" \
  --noise-strength 0.65 \     # Don't over-suppress — preserves weak signals
  --boost-db 12 \             # Aggressive speech band boost
  --whisper-model medium \    # Better accuracy on degraded audio
  --chunk-seconds 30          # Smaller chunks for more precise denoising
```

Key insight: **over-aggressive noise reduction destroys faint voices**. Use `preserve_weak_signals=true` and moderate `noise_reduction_strength` (0.5–0.7) for behind-the-door scenarios.

---

## Project Structure

```
audio-enhancer/
├── app/
│   ├── main.py              # FastAPI application
│   ├── core/
│   │   ├── pipeline.py      # Main processing engine (all DSP stages)
│   │   ├── command_parser.py # LLM + rule-based command interpretation
│   │   └── job_manager.py   # Thread-safe job tracking
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   └── utils/
│       ├── file_handler.py  # Streaming file I/O
│       └── model_loader.py  # Lazy ML model caching
├── scripts/
│   └── process_cli.py       # Command-line interface
├── ui/
│   └── index.html           # Web interface
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## License

MIT
