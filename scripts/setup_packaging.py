import PyInstaller.__main__
import os
from pathlib import Path

def build():
    # Define paths
    project_root = Path(__file__).parent.parent.resolve()
    src_dir = project_root / "src" / "spice"
    clamp_dir = src_dir / "clamp3"

    # Ensure tag embeddings exist
    if not (project_root / "tag_embeddings.npy").exists():
        print("Error: tag_embeddings.npy not found. Run autotagger.py first.")
        return

    # Change to project root for pyinstaller
    os.chdir(project_root)

    # PyInstaller arguments
    args = [
        str(project_root / "run.py"),           # Main script
        "--name=spice",                         # Executable name
        "--clean",                              # Clean cache
        "--noconfirm",                          # Replace output directory without asking
        "--windowed",                           # No console window (GUI mode)
        "--onedir",                             # Generate a directory (faster startup than onefile)

        # --- Data Files ---
        # Format: source_path;dest_path (Windows uses ;)

        # Include src/spice package
        f"--add-data={src_dir};spice",

        # Include Tag Embeddings
        f"--add-data=tag_embeddings.npy;.",

        # --- Hidden Imports ---
        # Libraries that PyInstaller often misses
        "--hidden-import=sklearn.neighbors._typedefs",
        "--hidden-import=sklearn.utils._cython_blas",
        "--hidden-import=sklearn.neighbors._quad_tree",
        "--hidden-import=sklearn.tree._utils",
        "--hidden-import=scipy.special.cython_special",
        "--hidden-import=scipy.linalg.cython_blas",
        "--hidden-import=scipy.linalg.cython_lapack",
        "--hidden-import=pandas",
        "--hidden-import=win32timezone",

        # Core modules
        "--hidden-import=spice.search",
        "--hidden-import=spice.indexer",
        "--hidden-import=spice.database",
        "--hidden-import=spice.autotagger",
        "--hidden-import=spice.waveform",
        "--hidden-import=spice.audio_analysis",

        # --- Excludes ---
        "--exclude-module=tkinter",
        "--exclude-module=matplotlib",
        "--exclude-module=notebook",
    ]

    print("Building spice...")
    PyInstaller.__main__.run(args)
    print("Build complete. Check 'dist/spice/spice.exe'")

if __name__ == "__main__":
    build()
