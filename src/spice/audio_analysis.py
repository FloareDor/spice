r"""
spice core tech validator (Librosa Edition)
Validates BPM and Key detection on a folder of WAV files.

Uses librosa instead of Essentia for cross-platform Windows support.

Usage:
    python validate_core.py "C:/path/to/samples"

Activate venv first:
    .\venv\Scripts\activate
"""
import argparse
import sys
import warnings
from pathlib import Path

# Filter librosa/scikit-learn warnings about n_fft being too large for short audio
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

try:
    import librosa
    import numpy as np
except ImportError:
    print("❌ Dependencies not installed. Run:")
    print("   .\\venv\\Scripts\\pip install librosa numpy")
    sys.exit(1)


# Key detection using Krumhansl-Schmuckler algorithm
# Major and minor key profiles (correlation weights)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def detect_key(y: np.ndarray, sr: int) -> tuple[str, str, float]:
    """
    Detect musical key using chroma features and Krumhansl-Schmuckler algorithm.
    
    Returns:
        tuple: (key_name, scale, confidence)
    """
    # Extract chroma features (12 pitch classes)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Normalize
    chroma_mean = chroma_mean / np.linalg.norm(chroma_mean)
    
    best_corr = -1
    best_key = 0
    best_scale = 'major'
    
    # Try all 12 keys for both major and minor
    for shift in range(12):
        # Rotate profiles to match each key
        major_rotated = np.roll(MAJOR_PROFILE, shift)
        minor_rotated = np.roll(MINOR_PROFILE, shift)
        
        # Normalize profiles
        major_norm = major_rotated / np.linalg.norm(major_rotated)
        minor_norm = minor_rotated / np.linalg.norm(minor_rotated)
        
        # Correlation
        major_corr = np.corrcoef(chroma_mean, major_norm)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, minor_norm)[0, 1]
        
        if major_corr > best_corr:
            best_corr = major_corr
            best_key = shift
            best_scale = 'major'
        
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = shift
            best_scale = 'minor'
    
    return KEY_NAMES[best_key], best_scale, best_corr


def analyze_audio(audio_path: str) -> dict:
    """
    Extract BPM, Key, and duration using librosa.
    
    Returns:
        dict with keys: bpm, key, scale, key_confidence, duration_sec
    """
    # Load audio (librosa auto-resamples to 22050Hz by default)
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    
    # Duration
    duration = librosa.get_duration(y=y, sr=sr)
    
    # BPM Detection using onset envelope + tempo estimation
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    
    # Handle tempo being an array (librosa versions vary)
    if hasattr(tempo, '__len__'):
        bpm = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        bpm = float(tempo)
    
    # Key Detection
    key, scale, key_confidence = detect_key(y, sr)
    
    return {
        "bpm": round(bpm, 1),
        "key": key,
        "scale": scale,
        "key_confidence": round(key_confidence, 2),
        "duration_sec": round(duration, 2),
        "num_beats": len(beat_frames)
    }


def main():
    parser = argparse.ArgumentParser(
        description="spice core tech validator - Analyze WAV files for BPM and Key"
    )
    parser.add_argument(
        "folder", 
        help="Path to folder containing WAV files"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search subfolders recursively"
    )
    args = parser.parse_args()
    
    folder = Path(args.folder)
    
    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        sys.exit(1)
    
    # Find WAV files (also support mp3, flac, etc - librosa handles many formats)
    extensions = ["*.wav", "*.WAV", "*.mp3", "*.MP3", "*.flac", "*.FLAC"]
    audio_files = []
    for ext in extensions:
        if args.recursive:
            audio_files.extend(folder.rglob(ext))
        else:
            audio_files.extend(folder.glob(ext))
    
    if not audio_files:
        print(f"❌ No audio files found in {folder}")
        sys.exit(1)
    
    # Header
    print()
    print("=" * 60)
    print("  spice core tech validator")
    print("  Librosa Audio Analysis (Cross-Platform)")
    print("=" * 60)
    print(f"  Folder: {folder}")
    print(f"  Found:  {len(audio_files)} audio file(s)")
    print("=" * 60)
    print()
    
    # Analyze each file
    success_count = 0
    error_count = 0
    
    for audio in sorted(audio_files):
        relative_path = audio.relative_to(folder) if args.recursive else audio.name
        print(f"📁 {relative_path}")
        
        try:
            info = analyze_audio(str(audio))
            
            # Format output
            key_display = f"{info['key']} {info['scale']}"
            confidence_pct = int(info['key_confidence'] * 100)
            
            print(f"   ⏱️  Duration: {info['duration_sec']}s ({info['num_beats']} beats)")
            print(f"   🥁 BPM:      {info['bpm']}")
            print(f"   🎹 Key:      {key_display} ({confidence_pct}% confidence)")
            print()
            
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()
            error_count += 1
    
    # Summary
    print("=" * 60)
    print(f"  ✅ Analyzed: {success_count} files")
    if error_count > 0:
        print(f"  ❌ Errors:   {error_count} files")
    print("=" * 60)
    print()
    print("🎉 Core tech validation complete!")
    print("   Next step: Add CLaMP 3 for semantic search embeddings")


if __name__ == "__main__":
    main()
