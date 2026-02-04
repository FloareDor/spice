"""
spice clamp 3 integration validator
Generates a test audio file and runs CLaMP 3 feature extraction.
"""
import os
import sys
import subprocess
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

def generate_test_audio(filename="test_audio.wav", duration=5.0):
    """Generates a simple sine wave audio file."""
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    y = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz A4
    sf.write(filename, y, sr)
    print(f"🎵 Generated test audio: {filename}")
    return filename

def run_clamp_extraction(input_dir, output_dir):
    """Runs the CLaMP 3 extraction script via subprocess."""
    clamp_script = Path("clamp3/clamp3_embd.py").resolve()
    
    if not clamp_script.exists():
        print(f"❌ CLaMP 3 script not found at {clamp_script}")
        return False
        
    cmd = [
        sys.executable, 
        str(clamp_script), 
        str(input_dir), 
        str(output_dir),
        "--get_global"
    ]
    
    print(f"🚀 Running CLaMP 3 extraction...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        # We need to set PYTHONPATH to include clamp3 directory so imports work
        env = os.environ.copy()
        # env["PYTHONPATH"] = str(clamp_script.parent) + os.pathsep + env.get("PYTHONPATH", "") # Not needed if we set cwd
        
        # Run from inside the clamp3 directory so relative 'preprocessing/audio' paths work
        # Input/Output paths must be absolute
        with open("validation_debug.log", "w", encoding="utf-8") as logf:
            result = subprocess.run(
                cmd, 
                cwd=str(clamp_script.parent),
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True, 
                check=True
            )
        print("✅ Extraction complete!")
        # print(result.stdout) <-- result.stdout is None if not captured
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Extraction failed!")
        # Output is already printed to console
        print(f"Exit code: {e.returncode}")
        print("Check validation_debug.log for details.")
        return False

def main():
    base_dir = Path.cwd()
    test_dir = base_dir / "test_data"
    output_dir = base_dir / "test_output"
    
    # Clean setup
    if test_dir.exists():
        for f in test_dir.glob("*"):
            f.unlink()
        test_dir.rmdir()
    if output_dir.exists():
        for f in output_dir.glob("*"):
            try:
                f.unlink()
            except:
                pass
        try:
            output_dir.rmdir() 
        except:
            pass
            
    test_dir.mkdir(exist_ok=True)
    
    # 1. Generate Audio
    wav_path = test_dir / "test_sine.wav"
    generate_test_audio(str(wav_path))
    
    # 2. Run Extraction
    success = run_clamp_extraction(test_dir, output_dir)
    
    if success:
        # 3. Verify Output
        # CLaMP 3 outputs .npy files. 
        # Structure might be output_dir/audio_features/test_sine.npy or just output_dir/test_sine.npy
        # based on utils.py logic: output_dir is passed directly.
        
        # Check files recursively
        npy_files = list(output_dir.rglob("*.npy"))
        
        if npy_files:
            print(f"✅ Found {len(npy_files)} embedding files:")
            for npy in npy_files:
                try:
                    data = np.load(npy)
                    print(f"   📄 {npy.name}: Shape {data.shape}")
                except Exception as e:
                    print(f"   ❌ Could not load {npy.name}: {e}")
        else:
            print("❌ No .npy files found in output directory!")
    
    # Cleanup
    # try:
    #     for f in test_dir.glob("*"): f.unlink()
    #     test_dir.rmdir()
    # except: pass

if __name__ == "__main__":
    main()
