import json
import os

import pytest

from rl import doctor, loadtest
from rl.doctor import ARENA, Check, Registry, check_disk, check_image, check_memory, setup_patch, write_lock

TASKS = ARENA / "tasks" / "tiktok"


@pytest.mark.parametrize("task", ["missing_user_id_in_guess_response", "video_url_format_validation"])
def test_setup_patch_extracts_a_reversible_diff(task):
    patch = setup_patch((TASKS / task / "setup.sh").read_text())
    assert patch.startswith("--- a/") and "+++ b/" in patch and patch.endswith("\n") and "EOF" not in patch


def test_no_patch_block_returns_none():
    assert setup_patch("#!/bin/bash\necho hi\n") is None


def test_registry_record_forget_and_dead_pid_only_stale(tmp_path, monkeypatch):
    reg = Registry(tmp_path / "p.jsonl")
    reg.record("la-a")
    reg.record("la-b")
    assert reg.names() == ["la-a", "la-b"]
    reg.forget("la-a")
    assert reg.names() == ["la-b"]
    monkeypatch.setattr(doctor, "project_objects", lambda name: dict(containers=["x"], networks=[], volumes=[]))
    assert reg.stale() == {}  # recorded by this (live) process
    (tmp_path / "p.jsonl").write_text(json.dumps(dict(project="la-dead", pid=999999, t=0)) + "\n")
    assert list(reg.stale()) == ["la-dead"]


def test_remove_project_refuses_foreign_names(tmp_path):
    reg = Registry(tmp_path / "p.jsonl")
    for name in ("qbittorrent", "torrent", "sonarr", "inspect-unrecorded"):
        with pytest.raises(ValueError):
            doctor.remove_project(name, reg)


def test_image_lock_mismatch_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "image_id", lambda env: "sha256:aaa")
    lock = tmp_path / "image.lock"
    assert check_image("tiktok", lock).status == "ok"
    write_lock("tiktok", lock)
    assert json.loads(lock.read_text()) == {"tiktok": "sha256:aaa"}
    monkeypatch.setattr(doctor, "image_id", lambda env: "sha256:bbb")
    c = check_image("tiktok", lock)
    assert c.status == "fail" and "image.lock" in c.detail
    monkeypatch.setattr(doctor, "image_id", lambda env: None)
    assert check_image("tiktok", lock).status == "fail"


def test_memory_gate_scales_with_live_stacks(monkeypatch):
    monkeypatch.setattr(doctor, "mem_available_gib", lambda: 3.0)
    monkeypatch.setattr(doctor, "live_stacks", lambda: 0)
    assert check_memory().status == "ok"
    assert check_memory(extra_stacks=2).status == "fail"  # 2.5 + 0.7 > 3.0
    monkeypatch.setattr(doctor, "live_stacks", lambda: 4)
    assert check_memory().status == "fail"


def test_network_pool_headroom_warns_under_ten(monkeypatch):
    class R:
        stdout = "\n".join(str(i) for i in range(25))
    monkeypatch.setattr(doctor, "docker", lambda *a, **k: R)
    monkeypatch.setattr(doctor, "network_pool_size", lambda: 31)
    c = doctor.check_networks()
    assert c.status == "warn" and c.data["free"] == 6
    monkeypatch.setattr(doctor, "network_pool_size", lambda: 28)
    assert doctor.check_networks(planned=4).status == "fail"
    monkeypatch.setattr(doctor, "network_pool_size", lambda: 270)
    assert doctor.check_networks().status == "ok"


def test_network_pool_size_reads_daemon_json(tmp_path, monkeypatch):
    p = tmp_path / "daemon.json"
    p.write_text(json.dumps({"default-address-pools": [{"base": "10.0.0.0/16", "size": 24}]}))
    real_path = doctor.Path
    monkeypatch.setattr(doctor, "Path", lambda x: p if x == "/etc/docker/daemon.json" else real_path(x))
    assert doctor.network_pool_size() == 256


def test_sampler_aborts_below_floor(monkeypatch):
    s = loadtest.Sampler(floor_gib=2.5)
    monkeypatch.setattr(loadtest, "mem_available_gib", lambda: 2.0)
    import asyncio

    async def go():
        t = asyncio.create_task(s.run())
        await asyncio.sleep(1.2)
        s.stop()
        await t
    asyncio.run(go())
    assert s.abort and s.low_gib <= 2.0


def test_loadtest_admission_skips_levels_and_writes_nothing_to_docker(monkeypatch, tmp_path):
    import argparse
    import asyncio
    monkeypatch.setattr(loadtest, "mem_available_gib", lambda: 6.0)
    monkeypatch.setattr(loadtest, "foreign_containers", lambda: [])
    args = argparse.Namespace(env="tiktok", task="video_url_format_validation", levels=[4, 8], floor=2.5, no_guard=True)
    out = asyncio.run(loadtest.main_async(args))
    assert all("skipped" in lv for lv in out["levels"]) and out["foreign_unchanged"]


def test_cgroup_helpers_read_files(tmp_path):
    (tmp_path / "memory.peak").write_text("12345\n")
    (tmp_path / "cpu.stat").write_text("usage_usec 777\nuser_usec 1\n")
    assert loadtest.read_int(tmp_path / "memory.peak") == 12345 and loadtest.read_int(tmp_path / "nope") is None
    assert loadtest.cpu_usec(tmp_path) == 777
