#!/bin/bash
set -euo pipefail
/workspace/venv-hf/bin/python vectors.py
/workspace/venv-hf/bin/python readout.py
