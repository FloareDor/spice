#!/usr/bin/env python
"""Quick launcher for spice"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from spice.app import main

if __name__ == "__main__":
    main()
