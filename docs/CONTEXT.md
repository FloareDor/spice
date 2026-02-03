# LocalVibe Project Context

## Project Overview
Building "LocalVibe" - a desktop sample library manager for music producers. Like Splice but 100% local/offline. Uses AI to search samples by "vibe" (semantic text) and DSP to detect BPM/Key.

**Master plan:** `e:\cmu-e\spice\docs\start.md`

## Tech Stack
- **Frontend:** Python + PyQt6 (for drag-to-DAW support)
- **Audio Analysis:** librosa (BPM, Key detection)
- **Semantic Search:** CLaMP 3 (text-to-audio embeddings)
- **Vector DB:** LanceDB
- **Metadata:** SQLite

## Current State (Phase 2 Started: CLaMP 3 Integrated ✅)

### Files Created
```
e:\cmu-e\spice\
├── docs/start.md          # Master plan document
├── validate_core.py       # BPM/Key detection script (working!)
├── validate_clamp.py      # CLaMP 3 embedding generation validation (working!)
├── requirements.txt       # Dependencies (librosa, torch, clamp3 reqs)
├── clamp3/                # Patched CLaMP 3 library (vendored)
└── venv/                  # Python virtual environment
```

### What Works
- **BPM/Key Detection**: `validate_core.py` analyzes audio using librosa.
- **Semantic Embeddings**: `validate_clamp.py` successfully generates 768-dim vectors from audio.
  - Generates test audio -> MERT features -> CLaMP 3 embedding.
  - Automatically downloads model weights on first run.

### Critical Patches (Windows Support)
- **`clamp3/utils.py`**: Modified to use `sys.executable` to ensure subprocesses use the venv python.
- **`clamp3/preprocessing/audio/MERT_utils.py`**: Replaced `torchaudio.load` with `librosa.load` to fix codec issues on Windows.

### How to Run
```powershell
cd e:\cmu-e\spice
.\venv\Scripts\activate

# Test BPM/Key
python validate_core.py "C:\path\to\samples"

# Test CLaMP 3 Embeddings
python validate_clamp.py
```

## Next Steps (Phase 2 Continued)
1.  **Store embeddings in LanceDB** for vector search
    - Schema: `filename (str)`, `embedding (vector[768])`, `metadata (json)`
    - Create `database.py` to handle localized vector storage.

2.  **Build PyQt6 UI** (Phase 3)
    - Search bar (text-to-audio via CLaMP 3)
    - Filter pills (BPM, Key)
    - Waveform preview
    - Drag-to-DAW support (QMimeData)

## Key Decisions Made
- **librosa over Essentia:** Essentia doesn't have Windows Python bindings.
- **Patched CLaMP 3:** Vendored the library to apply Windows-specific patches (librosa substitution).
- **venv:** Using Python venv for this project.
