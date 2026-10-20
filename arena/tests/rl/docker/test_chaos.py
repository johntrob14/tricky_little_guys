"""Real-container fault tests for the infrastructure layer. Only objects named la-test-* are created."""
import asyncio
import json
import os
import stat
import subprocess
import time

import pytest

from rl.classify import AGENT, INFRA, classify_exception, classify_text
from rl.doctor import (ARENA, ComposeSandbox, DockerTimeout, Registry, check_daemon, check_networks, check_stale, container_state, docker, docker_retry, foreign_containers,
                       project_objects, remove_project)

pytestmark = pytest.mark.docker
IMAGE = "linuxarena/env-tiktok:latest"
COMPOSE = """services:
  default:
    image: linuxarena/env-tiktok:latest
    x-local: true
    stop_grace_period: 1s
    command: sleep infinity
networks:
  default:
    internal: true
"""


@pytest.fixture
def reg(tmp_path):
    return Registry(tmp_path / "projects.jsonl")


@pytest.fixture
def sbx(tmp_path, reg):
    project = f"la-test-chaos-{os.getpid()}"
    (tmp_path / "compose.yml").write_text(COMPOSE)
    reg.record(project)
    sb = ComposeSandbox(project, tmp_path / "compose.yml")
    subprocess.run(sb._base() + ["up", "-d", "--quiet-pull"], check=True, capture_output=True, timeout=120)
    yield sb
    remove_project(project, reg)
    assert not any(project_objects(project).values())


def cid(sb):
    return docker("ps", "-aq", "--filter", f"label=com.docker.compose.project={sb.project}").stdout.split()[0]


def test_daemon_answers_fast():
    c = check_daemon()
    assert c.status == "ok" and c.seconds < 5


def test_exec_works_then_container_killed_is_container_lost(sbx):
    assert asyncio.run(sbx.exec(["echo", "hi"])).stdout.strip() == "hi"
    docker("kill", cid(sbx))
    r = asyncio.run(sbx.exec(["echo", "hi"]))
    assert not r.success
    f = classify_text(r.stderr, "exec", container_state(cid(sbx)))
    assert (f.failure_class, f.reason) == (INFRA, "container_lost")


def test_container_removed_midway_is_container_lost(sbx):
    c = cid(sbx)
    docker("rm", "-f", c)
    r = asyncio.run(sbx.exec(["echo", "hi"]))
    assert not r.success and container_state(c) == "gone"
    assert classify_text(r.stderr, "exec", "gone").reason == "container_lost"


def test_paused_container_exec_fails_fast_and_is_infra(sbx):
    docker("pause", cid(sbx))
    r = asyncio.run(sbx.exec(["echo", "hi"], timeout=5))
    state = container_state(cid(sbx))
    docker("unpause", cid(sbx))
    assert not r.success and state == "paused"
    assert classify_text(r.stderr, "exec", state).failure_class == INFRA


def test_hung_docker_cli_is_killed_on_the_host_and_is_exec_hang(sbx, tmp_path, monkeypatch):
    shim = tmp_path / "bin" / "docker"
    shim.parent.mkdir()
    real = subprocess.run(["which", "docker"], capture_output=True, text=True).stdout.strip()
    shim.write_text(f'#!/bin/bash\nif [[ "$*" == *" exec "* ]]; then sleep 120; fi\nexec {real} "$@"\n')
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{shim.parent}:{os.environ['PATH']}")
    t0 = time.monotonic()
    with pytest.raises(TimeoutError) as e:
        asyncio.run(sbx.exec(["echo", "hi"], timeout=1))
    assert time.monotonic() - t0 < 30
    monkeypatch.setenv("PATH", os.environ["PATH"].split(":", 1)[1])
    f = classify_exception(e.value, "exec", None)
    assert (f.failure_class, f.reason) == (INFRA, "exec_hang")


def test_slow_agent_command_in_running_container_is_agent_timeout(sbx):
    with pytest.raises(TimeoutError) as e:
        asyncio.run(sbx.exec(["sleep", "30"], timeout=2))
    f = classify_exception(e.value, "exec", container_state(cid(sbx)))
    assert (f.failure_class, f.reason) == (AGENT, "tool_timeout")


def test_name_conflict_is_classified():
    name = f"la-test-conflict-{os.getpid()}"
    try:
        assert docker("create", "--name", name, "hello-world").returncode == 0
        r = docker("create", "--name", name, "hello-world")
        assert r.returncode != 0
        assert classify_text(r.stderr).reason == "name_conflict"
    finally:
        docker("rm", "-f", name)


def test_oom_with_tiny_cap_is_oom_killed():
    name = f"la-test-oom-{os.getpid()}"
    try:
        r = docker("run", "--name", name, "--memory", "32m", "--memory-swap", "32m", "--network", "none", "--entrypoint", "python3", IMAGE, "-c", "x = bytearray(400*1024*1024); print(len(x))", timeout=60)
        assert r.returncode == 137
        assert container_state(name) == "oomkilled"
        assert classify_text("agent container died", "exec", container_state(name)).reason == "oom_killed"
    finally:
        docker("rm", "-f", name)


def test_missing_image_is_image_missing():
    name = f"la-test-noimage-{os.getpid()}"
    r = docker("create", "--name", name, "linuxarena/env-does-not-exist:latest", timeout=60)
    docker("rm", "-f", name)
    assert r.returncode != 0 and classify_text(r.stderr).reason == "image_missing"


def make_shim(tmp_path, body):
    shim = tmp_path / "bin" / "docker"
    shim.parent.mkdir()
    real = subprocess.run(["which", "docker"], capture_output=True, text=True).stdout.strip()
    shim.write_text(f"#!/bin/bash\n{body.replace('REAL', real)}\nexec {real} \"$@\"\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    return shim.parent


def test_path_shim_transient_errors_are_retried_for_idempotent_calls(tmp_path, monkeypatch):
    counter = tmp_path / "count"
    counter.write_text("0")
    body = f'n=$(cat {counter}); echo $((n+1)) > {counter}\nif [ "$n" -lt 2 ]; then echo "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?" >&2; exit 1; fi'
    monkeypatch.setenv("PATH", f"{make_shim(tmp_path, body)}:{os.environ['PATH']}")
    r = docker_retry("ps", "-q", backoff=0.01)
    assert r.returncode == 0 and counter.read_text().strip() == "3"
    counter.write_text("0")
    plain = docker("ps", "-q")
    assert plain.returncode == 1 and classify_text(plain.stderr).reason == "docker_daemon_unreachable"


def test_path_shim_slow_docker_raises_docker_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", f"{make_shim(tmp_path, 'sleep 20')}:{os.environ['PATH']}")
    t0 = time.monotonic()
    with pytest.raises(DockerTimeout):
        docker("ps", timeout=1)
    assert time.monotonic() - t0 < 5
    c = check_daemon(limit=1)
    assert c.status == "fail" and "exceeded" in c.detail


def test_stale_project_cleanup_removes_exactly_the_recorded_name(reg):
    mine, other = f"la-test-stale-{os.getpid()}", f"la-test-bystander-{os.getpid()}"
    lbl = "com.docker.compose.project"
    try:
        for p in (mine, other):
            docker("create", "--name", f"{p}-default-1", "--label", f"{lbl}={p}", "hello-world")
        reg.path.write_text(json.dumps(dict(project=mine, pid=999999, t=0)) + "\n")
        assert list(reg.stale()) == [mine]
        assert check_stale(False, reg).status == "fail"
        done = check_stale(True, reg)
        assert done.status == "ok"
        assert not any(project_objects(mine).values()) and any(project_objects(other).values())
        with pytest.raises(ValueError):
            remove_project("qbittorrent", reg)
        with pytest.raises(ValueError):
            remove_project("inspect-notrecorded", reg)
    finally:
        for p in (mine, other):
            docker("rm", "-f", f"{p}-default-1")


def test_live_registered_pid_is_not_stale(reg):
    p = f"la-test-live-{os.getpid()}"
    try:
        docker("create", "--name", f"{p}-default-1", "--label", f"com.docker.compose.project={p}", "hello-world")
        reg.record(p)
        assert reg.stale() == {}
    finally:
        docker("rm", "-f", f"{p}-default-1")


def test_network_pool_headroom_counts_networks():
    c = check_networks()
    assert c.data["used"] == int(docker("network", "ls", "-q").stdout.count("\n")) and c.status in ("ok", "warn")
