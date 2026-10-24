#!/usr/bin/env python3
"""Root CLI entry point forwarding arguments to src/main.py.

Usage:
    python Ex0.py <file.nrrd> <file.csv> <file.mp4>
"""

import sys
from src.main import main

if __name__ == "__main__":
    sys.exit(main())
