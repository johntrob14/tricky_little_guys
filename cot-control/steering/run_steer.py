"""Launcher for steer.py with a __main__ guard: with tensor parallelism vLLM spawns workers, which re-import the main
module; steer.py is a flat script, so it is run from here instead of being the main module."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.argv[0] = str(Path(__file__).with_name("steer.py"))
    runpy.run_path(sys.argv[0], run_name="steer_main")
