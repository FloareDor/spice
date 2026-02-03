import PyInstaller.__main__
import os
from pathlib import Path

def build():
    # Define paths
    base_dir = Path(__file__).parent.resolve()
    clamp_dir = base_dir / "clamp3"
    
    # Ensure tag embeddings exist
    if not (base_dir / "tag_embeddings.npy").exists():
        print("Error: tag_embeddings.npy not found. Run autotagger.py first.")
        return

    # PyInstaller arguments
    args = [
        "app.py",                           # Main script
        "--name=LocalVibe",                 # Executable name
        "--clean",                          # Clean cache
        "--noconfirm",                      # Replace output directory without asking
        "--windowed",                       # No console window (GUI mode)
        "--onedir",                         # Generate a directory (faster startup than onefile)
        
        # --- Data Files ---
        # Format: source_path;dest_path (Windows uses ;)
        
        # Include CLaMP 3 folder
        f"--add-data={clamp_dir};clamp3",
        
        # Include Tag Embeddings
        f"--add-data=tag_embeddings.npy;.",
        
        # Include LanceDB (if localvibe.lance exists, purely optional as user might want fresh start)
        # But we definitely need the schema if it's code-defined.
        # Actually, let's NOT include the DB. The app creates it if missing.
        
        # --- Hidden Imports ---
        # Libraries that PyInstaller often misses
        "--hidden-import=sklearn.neighbors._typedefs",
        "--hidden-import=sklearn.utils._cython_blas",
        "--hidden-import=sklearn.neighbors._quad_tree",
        "--hidden-import=sklearn.tree._utils",
        "--hidden-import=scipy.special.cython_special",
        "--hidden-import=scipy.linalg.cython_blas",
        "--hidden-import=scipy.linalg.cython_lapack",
        "--hidden-import=pandas", # Often needed by lance/sklearn
        "--hidden-import=win32timezone", # Common Windows issue
        
        # Core modules (just to be safe)
        "--hidden-import=search",
        "--hidden-import=indexer",
        "--hidden-import=database",
        "--hidden-import=autotagger",
        "--hidden-import=waveform",
        "--hidden-import=validate_core",
        
        # --- Excludes ---
        # Reduce size by excluding unneeded heavy libraries if possible
        # (Be careful here, only exclude what we KNOW isn't used)
        "--exclude-module=tkinter",
        "--exclude-module=matplotlib",
        "--exclude-module=notebook",
        
        # --- Icon ---
        # (Optional: Add if we had an icon)
        # "--icon=resources/icon.ico"
    ]

    print("Building LocalVibe...")
    PyInstaller.__main__.run(args)
    print("Build complete. Check 'dist/LocalVibe/LocalVibe.exe'")

if __name__ == "__main__":
    build()
