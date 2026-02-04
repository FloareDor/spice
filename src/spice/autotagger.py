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

from .model_wrapper import ModelWrapper

def generate_tag_embeddings(output_file: str = TAG_EMBEDDINGS_FILE):
    """
    Generate embeddings for all tags using CLaMP 3 and save to file.
    """
    print("Generating tag embeddings...")
    
    # Initialize ModelWrapper (will reuse existing weights/code)
    model = ModelWrapper(quantize=True)
    
    embeddings = {}
    
    # Generate embeddings
    # Process in batches if list is huge, but here it's small (~20 tags)
    vectors = model.compute_text_embeddings(TAGS)
    
    for tag, vec in zip(TAGS, vectors):
        embeddings[tag] = vec
    
    # Save as dictionary
    np.save(output_file, embeddings)
    
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
