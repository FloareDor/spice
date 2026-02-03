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

## Current State (Phase 1 Complete ✅)

### Files Created
```
e:\cmu-e\spice\
├── docs/start.md          # Master plan document
├── validate_core.py       # BPM/Key detection script (working!)
├── requirements.txt       # Dependencies (librosa)
└── venv/                  # Python virtual environment
```

### What Works
- `validate_core.py` analyzes audio files for BPM and Key
- Uses librosa (cross-platform, works on Windows unlike Essentia)
- Supports WAV, MP3, FLAC
- Key detection uses Krumhansl-Schmuckler algorithm

### How to Run
```powershell
cd e:\cmu-e\spice
.\venv\Scripts\activate
python validate_core.py "C:\path\to\samples"
```

## Next Steps (Phase 2)
1. **Add CLaMP 3** for semantic audio embeddings
   - Repo: https://github.com/sanderwood/clamp3
   - Use "CLaMP 3 SAAS" model for audio retrieval
   - Requires PyTorch, MERT audio encoder (~1GB model)

2. **Store embeddings in LanceDB** for vector search

3. **Build PyQt6 UI** with:
   - Search bar (text-to-audio via CLaMP 3)
   - Filter pills (BPM, Key)
   - Waveform preview
   - Drag-to-DAW support (QMimeData)

## Key Decisions Made
- **librosa over Essentia:** Essentia doesn't have Windows Python bindings
- **Phased approach:** Get BPM/Key working first, add CLaMP 3 later
- **venv:** Using Python venv for this project
