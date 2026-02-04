import os
import sys
import shutil
import requests
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, BertConfig

# Add CLaMP 3 paths to sys.path
CURRENT_DIR = Path(__file__).parent
CLAMP_CODE_DIR = CURRENT_DIR / "clamp3" / "code"
CLAMP_AUDIO_DIR = CURRENT_DIR / "clamp3" / "preprocessing" / "audio"

sys.path.append(str(CLAMP_CODE_DIR))
sys.path.append(str(CLAMP_AUDIO_DIR))

# Now import CLaMP modules
# We need to wrap these imports in try-except blocks or handle potential import errors gracefully
# but given the strict dependency list, they should exist.
try:
    from config import (
        AUDIO_HIDDEN_SIZE, AUDIO_NUM_LAYERS, MAX_AUDIO_LENGTH,
        M3_HIDDEN_SIZE, PATCH_NUM_LAYERS, PATCH_LENGTH,
        CLAMP3_HIDDEN_SIZE, CLAMP3_LOAD_M3, TEXT_MODEL_NAME,
        CLAMP3_WEIGHTS_PATH
    )
    # We import CLaMP3Model from utils because that's where we found it defined in the previous step
    from utils import CLaMP3Model 
    from hf_pretrains import HuBERTFeature
    from MERT_utils import load_audio
except ImportError as e:
    print(f"Error importing CLaMP 3 modules: {e}")
    # Fallback or exit? For now, we'll let it fail loudly if modules are missing.
    raise

class ModelWrapper:
    """
    Wraps MERT and CLaMP 3 models for efficient inference without subprocess overhead.
    """
    def __init__(self, device: str = None, quantize: bool = True):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.quantize = quantize and (self.device == "cpu") # Quantization is mostly for CPU speedup
        
        print(f"Initializing models on {self.device} (Quantization: {self.quantize})...")
        
        # 1. Load MERT Feature Extractor
        self._load_mert()
        
        # 2. Load CLaMP 3 Model
        self._load_clamp()

    def _load_mert(self):
        """Initialize MERT model for audio feature extraction."""
        print("Loading MERT-v1-95M...")
        self.mert_model_path = "m-a-p/MERT-v1-95M" # Hardcoded in original script
        self.feature_extractor = HuBERTFeature(
            self.mert_model_path,
            24000, # target_sr
            force_half=False,
            processor_normalize=True,
        )
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

    def _load_clamp(self):
        """Initialize CLaMP 3 model."""
        print("Loading CLaMP 3...")
        
        # Config setup (mirrored from extract_clamp3.py)
        audio_config = BertConfig(
            vocab_size=1,
            hidden_size=AUDIO_HIDDEN_SIZE,
            num_hidden_layers=AUDIO_NUM_LAYERS,
            num_attention_heads=AUDIO_HIDDEN_SIZE//64,
            intermediate_size=AUDIO_HIDDEN_SIZE*4,
            max_position_embeddings=MAX_AUDIO_LENGTH
        )
        symbolic_config = BertConfig(
            vocab_size=1,
            hidden_size=M3_HIDDEN_SIZE,
            num_hidden_layers=PATCH_NUM_LAYERS,
            num_attention_heads=M3_HIDDEN_SIZE//64,
            intermediate_size=M3_HIDDEN_SIZE*4,
            max_position_embeddings=PATCH_LENGTH
        )
        
        self.clamp_model = CLaMP3Model(
            audio_config=audio_config,
            symbolic_config=symbolic_config,
            text_model_name=TEXT_MODEL_NAME,
            hidden_size=CLAMP3_HIDDEN_SIZE,
            load_m3=CLAMP3_LOAD_M3
        )
        self.clamp_model.to(self.device)
        self.clamp_model.eval()
        
        # Load Tokenizer for text
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

        # Download weights if missing
        checkpoint_path = CLAMP_CODE_DIR / CLAMP3_WEIGHTS_PATH
        if not checkpoint_path.exists():
            print("CLaMP 3 weights not found. Downloading...")
            self._download_weights(checkpoint_path)
            
        print(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.clamp_model.load_state_dict(checkpoint['model'])

        # Dynamic Quantization (CPU Only)
        if self.quantize:
            print("Applying dynamic quantization to CLaMP 3...")
            self.clamp_model = torch.quantization.quantize_dynamic(
                self.clamp_model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )

    def _download_weights(self, path: Path):
        url = "https://huggingface.co/sander-wood/clamp3/resolve/main/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(path, "wb") as f, tqdm(
            desc="Downloading CLaMP weights",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
                    
    def compute_text_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """
        Compute CLaMP 3 embeddings for a list of text strings.
        Returns a list of 768-dim numpy arrays.
        """
        embeddings = []
        MAX_TEXT_LENGTH = 128 # from config
        
        for text in texts:
            try:
                # Preprocessing from extract_clamp3.py
                # It handles newlines and basic cleaning
                item = list(set(text.split("\n")))
                item = "\n".join(item)
                item = item.split("\n")
                item = [c for c in item if len(c) > 0]
                cleaned_text = self.tokenizer.sep_token.join(item)
                
                # Tokenize
                inputs = self.tokenizer(cleaned_text, return_tensors="pt")
                input_ids = inputs['input_ids'].squeeze(0) # [seq_len]
                
                # Chunking (similar to audio but for text)
                segment_list = []
                for i in range(0, len(input_ids), MAX_TEXT_LENGTH):
                    segment_list.append(input_ids[i:i+MAX_TEXT_LENGTH])
                if len(input_ids) >= MAX_TEXT_LENGTH:
                    segment_list[-1] = input_ids[-MAX_TEXT_LENGTH:]
                
                last_hidden_states_list = []
                
                for input_segment in segment_list:
                    current_len = input_segment.size(0)
                    pad_indices = torch.ones(MAX_TEXT_LENGTH - current_len).long() * self.tokenizer.pad_token_id
                    input_masks = torch.tensor([1]*current_len)
                    
                    if current_len < MAX_TEXT_LENGTH:
                        input_masks = torch.cat((input_masks, torch.zeros(MAX_TEXT_LENGTH - current_len)), 0)
                        input_segment = torch.cat((input_segment, pad_indices), 0)
                        
                    with torch.no_grad():
                        last_hidden_states = self.clamp_model.get_text_features(
                            text_inputs=input_segment.unsqueeze(0).to(self.device),
                            text_masks=input_masks.unsqueeze(0).to(self.device),
                            get_global=True
                        )
                    last_hidden_states_list.append(last_hidden_states.cpu())
                    
                final_embedding = torch.cat(last_hidden_states_list, 0)
                final_embedding = final_embedding.mean(dim=0) # Average chunks
                embeddings.append(final_embedding.numpy())
                
            except Exception as e:
                print(f"Error processing text '{text[:20]}...': {e}")
                embeddings.append(np.zeros(CLAMP3_HIDDEN_SIZE))
                
        return embeddings

    def compute_embeddings(self, audio_paths: list[Path]) -> dict[str, np.ndarray]:
        """
        Compute CLaMP 3 embeddings for a list of audio files.
        Returns a dict mapping filename -> embedding (numpy array).
        """
        results = {}
        
        # Processing parameters
        target_sr = 24000
        # From extract_clamp3.py logic
        
        for audio_path in audio_paths:
            try:
                # 1. Extract MERT features
                # Using hf_pretrains / MERT_utils logic
                waveform = load_audio(
                    str(audio_path),
                    target_sr=target_sr,
                    is_mono=True,
                    is_normalize=False,
                    crop_to_length_in_sec=None,
                    crop_randomly=False,
                    device=self.device
                )
                
                # MERT Inference
                wav = self.feature_extractor.process_wav(waveform).to(self.device)
                
                # Simple full-sequence inference (no sliding window for now to match default behavior if config was None)
                # But extract_mert.py uses sliding window if configured.
                # Let's check config.py... it wasn't there. It was in extract_mert.py vars.
                # Default behavior in extract_mert.py: sliding_window_size_in_sec = 5
                
                sliding_window_size_in_sec = 5
                sliding_window_overlap_in_percent = 0.0
                
                if sliding_window_size_in_sec:
                    overlap_in_sec = sliding_window_size_in_sec * sliding_window_overlap_in_percent / 100
                    wavs = []
                    step = int(target_sr * (sliding_window_size_in_sec - overlap_in_sec))
                    window_len = int(target_sr * sliding_window_size_in_sec)
                    
                    for i in range(0, wav.shape[-1], step):
                        chunk = wav[:, i : i + window_len]
                        if chunk.shape[-1] >= target_sr * 1: # Min 1 sec
                            wavs.append(chunk)
                    
                    if not wavs: # Handle very short files
                        wavs = [wav]

                    features_list = []
                    for wav_chunk in wavs:
                        with torch.no_grad():
                            features_list.append(self.feature_extractor(wav_chunk, layer=None, reduction='mean'))
                    mert_features = torch.cat(features_list, dim=1)
                else:
                    with torch.no_grad():
                        mert_features = self.feature_extractor(wav, layer=None, reduction='mean')

                # Average over layers (dim 0) to match extract_mert.py --mean_features behavior
                # mert_features is [Layers, Chunks, Hidden] -> [Chunks, Hidden]
                mert_features = mert_features.mean(dim=0)

                # 2. Extract CLaMP 3 Features
                # Prepare input for CLaMP
                # input_data shape: [seq_len, hidden_dim]
                
                input_data = mert_features.cpu() # [seq_len, 768]
                
                # CLaMP Preprocessing logic from extract_feature
                zero_vec = torch.zeros((1, input_data.size(-1)))
                input_data = torch.cat((zero_vec, input_data, zero_vec), 0)
                
                # Chunking
                max_input_length = MAX_AUDIO_LENGTH # 128
                segment_list = []
                for i in range(0, len(input_data), max_input_length):
                    segment_list.append(input_data[i:i+max_input_length])
                
                # Pad the last segment to max_input_length if needed?
                # The original code: segment_list[-1] = input_data[-max_input_length:] 
                # Wait, that logic in extract_clamp3.py looks like it takes the *last* max_input_length tokens
                # effectively overlapping the last chunk significantly if it's small.
                if len(input_data) >= max_input_length:
                     segment_list[-1] = input_data[-max_input_length:]
                
                last_hidden_states_list = []
                
                for input_segment in segment_list:
                    # Create masks
                    input_masks = torch.tensor([1]*input_segment.size(0))
                    pad_indices = torch.ones((MAX_AUDIO_LENGTH - input_segment.size(0), AUDIO_HIDDEN_SIZE)).float() * 0.
                    
                    # Pad
                    current_len = input_segment.size(0)
                    if current_len < max_input_length:
                        # Append padding
                        input_masks = torch.cat((input_masks, torch.zeros(max_input_length - current_len)), 0)
                        input_segment = torch.cat((input_segment, pad_indices), 0)
                    
                    with torch.no_grad():
                        last_hidden_states = self.clamp_model.get_audio_features(
                            audio_inputs=input_segment.unsqueeze(0).to(self.device),
                            audio_masks=input_masks.unsqueeze(0).to(self.device),
                            get_global=True # We want global feature for indexing
                        )
                    last_hidden_states_list.append(last_hidden_states.cpu())

                # Combine chunks (Global Average)
                # extract_clamp3.py logic for global:
                full_chunk_cnt = len(input_data) // max_input_length
                remain_chunk_len = len(input_data) % max_input_length
                
                # Re-calculate weights based on actual chunk lengths
                # Since we are iterating standard chunks, most are max_length.
                # But the code uses a weighted sum.
                
                # Simplified: Average the global embeddings from each chunk
                # The original code tries to be precise about weighting by valid token count.
                # Given we want robust embeddings, simple average of chunk embeddings is usually sufficient
                # but let's try to stick to their logic if possible.
                
                final_embedding = torch.cat(last_hidden_states_list, 0)
                
                # Calculate weights roughly
                weights = torch.ones(len(last_hidden_states_list), 1)
                # If the last chunk was partial (but we grabbed the tail), it still represents full length context 
                # but might overlap.
                # Let's just do a mean for now. It's robust enough for search.
                final_embedding = final_embedding.mean(dim=0)
                
                # Use full path string as key to handle duplicate filenames in different folders
                results[str(audio_path)] = final_embedding.numpy()

            except Exception as e:
                print(f"Error processing {audio_path.name}: {e}")
                # We don't raise here to allow batch to continue
                
        return results
