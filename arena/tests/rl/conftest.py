import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "tinker: needs the live Tinker API (spends money); skipped unless ARENA_RL_RUN_TINKER=1")


def pytest_collection_modifyitems(config, items):
    import os

    if os.environ.get("ARENA_RL_RUN_TINKER") == "1":
        return
    skip = pytest.mark.skip(reason="live Tinker test; set ARENA_RL_RUN_TINKER=1 (John's go required)")
    for item in items:
        if "tinker" in item.keywords:
            item.add_marker(skip)
