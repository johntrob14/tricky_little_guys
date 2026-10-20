import os

import pytest

from rl.doctor import foreign_containers


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: creates la-* docker objects on this host; skipped unless ARENA_RL_DOCKER=1")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("ARENA_RL_DOCKER") == "1":
        return
    skip = pytest.mark.skip(reason="docker test; set ARENA_RL_DOCKER=1")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def foreign_untouched():
    """Every docker test leaves the non-la containers (John's services) exactly as they were."""
    before = foreign_containers()
    yield
    assert foreign_containers() == before
