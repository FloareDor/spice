import numpy as np
import subprocess
import sys
import shutil
from pathlib import Path
import json
import os

# Standard producer tags
TAGS = [
    # Instruments
    "Kick", "Snare", "Hi-Hat", "Clap", "Percussion", "Cymbal", "Tom",
    "Bass", "808", "Synth", "Pad", "Lead", "Pluck", "Piano", "Guitar",
    "Strings", "Brass", "Vocals", "FX", "Foley", "Texture", "Drone",
    
    # Genres
    "Trap", "Hip Hop", "House", "Techno", "Dubstep", "DnB", "Lo-Fi",
    "Cinematic", "Pop", "RnB", "Rock", "Jazz", "Ambient",
    
    # Characteristics
    "Dark", "Bright", "Warm", "Cold", "Heavy", "Soft", "Punchy",
    "Distorted", "Clean", "Dry", "Wet", "Aggressive", "Chill",
    "Melodic", "Rhythmic", "One Shot", "Loop"
]

TAG_EMBEDDINGS_FILE = "tag_embeddings.npy"

def generate_tag_embeddings(output_file: str = TAG_EMBEDDINGS_FILE):
    """
    Generate embeddings for all tags using CLaMP 3 and save to file.
    """
    print("Generating tag embeddings...")
    clamp_script = Path("clamp3/clamp3_embd.py").resolve()
    if not clamp_script.exists():
        raise FileNotFoundError(f"CLaMP 3 not found at {clamp_script}")

    temp_dir = Path(".tag_gen_temp")
    temp_dir.mkdir(exist_ok=True)
    
    temp_input = temp_dir / "input"
    temp_input.mkdir(exist_ok=True)
    temp_output = temp_dir / "output"
    
    # Create a text file for each tag
    # CLaMP script expects input files. For text, it might expect .txt
    # We will use the same method as search.py: write text files
    
    tag_map = {} # filename -> tag
    
    for i, tag in enumerate(TAGS):
        safe_tag = "".join([c if c.isalnum() else "_" for c in tag])
        fname = f"{i:03d}_{safe_tag}.txt"
        with open(temp_input / fname, "w", encoding="utf-8") as f:
            f.write(tag)
        tag_map[fname] = tag

    # Run CLaMP
    cmd = [
        sys.executable,
        str(clamp_script),
        str(temp_input.resolve()),
        str(temp_output.resolve()),
        "--get_global"
    ]

    result = subprocess.run(
        cmd,
        cwd=str(clamp_script.parent),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"CLaMP failed: {result.returncode}")

    # Collect embeddings
    embeddings = {} # tag -> vector
    
    for npy_file in temp_output.rglob("*.npy"):
        stem = npy_file.stem
        # clamp script might rename files slightly or keep stem
        # We need to map back to tag. 
        # The script outputs {filename}.npy
        
        # Find original tag
        original_fname = f"{stem}.txt"
        if original_fname in tag_map:
            tag = tag_map[original_fname]
            vec = np.load(npy_file)
            if vec.ndim > 1:
                vec = vec.flatten()
            embeddings[tag] = vec
    
    # Save as dictionary
    np.save(output_file, embeddings)
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Saved {len(embeddings)} tag embeddings to {output_file}")
    return embeddings

def load_tag_embeddings(file_path: str = TAG_EMBEDDINGS_FILE):
    """Load tag embeddings from file, generating them if needed."""
    if not os.path.exists(file_path):
        return generate_tag_embeddings(file_path)
    
    # Allow loading pickled dictionary
    return np.load(file_path, allow_pickle=True).item()

def suggest_tags(audio_embedding: np.ndarray, tag_embeddings: dict, top_k: int = 5, threshold: float = 0.0) -> list[str]:
    """
    Suggest tags for an audio embedding based on cosine similarity.
    """
    scores = []
    
    # Normalize input
    audio_norm = np.linalg.norm(audio_embedding)
    if audio_norm == 0:
        return []
    
    for tag, tag_emb in tag_embeddings.items():
        tag_norm = np.linalg.norm(tag_emb)
        if tag_norm == 0:
            continue
            
        # Cosine similarity
        sim = np.dot(audio_embedding, tag_emb) / (audio_norm * tag_norm)
        
        if sim > threshold:
            scores.append((tag, sim))
            
    # Sort by score desc
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return [tag for tag, score in scores[:top_k]]

if __name__ == "__main__":
    # Test generation
    generate_tag_embeddings()
