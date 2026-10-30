"""Preflight for RL runs, no Tinker: `python -m rl.doctor [--env tiktok] [--no-golden] [--cleanup] [--write-lock] [--json out.json]`.

Checks: docker daemon answers in <5s; image present (and equal to rl_data/image.lock when that exists); disk and MemAvailable gates; stale compose
projects of ours (recorded in ~/.cache/la/projects.jsonl; removal is by exact recorded name only); docker network pool headroom; and a golden control
per task: the bug patch from setup.sh must make tests.py fail, its reverse must make it pass, in a Docker Compose project la-doctor-<pid> built from
the vendored compose.yml (no Inspect, torn down by exact name). Score.py tasks get a wrong-answer smoke check on the pristine tree, answer.txt tasks a
parse check. Exit status 0 iff nothing failed.

Also home of the docker helpers the other rl modules share: docker(), Registry, ComposeSandbox (the async exec/write_file/read_file subset of Inspect's
SandboxEnvironment), Stack, foreign_containers().
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ARENA = Path(__file__).resolve().parent.parent
RL_DATA = ARENA / "rl_data"
CACHE = Path(os.environ.get("LA_CACHE", Path.home() / ".cache" / "la"))
REGISTRY = CACHE / "projects.jsonl"
GIB = 1024 ** 3
BASE_MEM_GIB, PER_STACK_GIB = 2.5, 0.35
NET_POOL_DEFAULT = 31
NET_FREE_WARN = 10
OUR_PREFIXES = ("la-", "inspect-")


class DockerTimeout(Exception):
    pass


def docker(*args, timeout: float = 30, input: bytes | str | None = None) -> subprocess.CompletedProcess:
    """Run `docker <args>`; never raises on a nonzero exit, raises DockerTimeout (child killed) when it hangs."""
    try:
        return subprocess.run(["docker", *args], capture_output=True, text=isinstance(input, str) or input is None, input=input, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise DockerTimeout(f"docker {' '.join(map(str, args[:3]))} exceeded {timeout}s") from e


def docker_retry(*args, tries: int = 3, backoff: float = 1.0, timeout: float = 30, ok_codes=(0,)) -> subprocess.CompletedProcess:
    """For idempotent calls only (inspect, ps, cp, rm, network ls): retry a failed or hung call `tries` times. Never use for `exec` of agent commands."""
    last = None
    for i in range(tries):
        try:
            last = docker(*args, timeout=timeout)
            if last.returncode in ok_codes:
                return last
        except DockerTimeout as e:
            last = e
        time.sleep(backoff * (i + 1) if i + 1 < tries else 0)
    if isinstance(last, DockerTimeout):
        raise last
    return last


def container_state(container: str) -> str:
    """running | paused | oomkilled | exited | gone, from `docker inspect` (retried); the input classify_text wants as container_running."""
    r = docker_retry("inspect", "--format", "{{.State.Status}} {{.State.OOMKilled}}", container)
    if r.returncode:
        return "gone"
    status, oom = r.stdout.split()
    return "oomkilled" if oom == "true" else status if status in ("running", "paused") else "exited"


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2
    raise RuntimeError("no MemAvailable")


def project_containers(project: str) -> list[dict]:
    """Containers of the compose project with exactly this name (label equality, not a name match)."""
    out = docker("ps", "-a", "--no-trunc", "--filter", f"label=com.docker.compose.project={project}", "--format", "{{json .}}").stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def live_stacks() -> int:
    """Compose projects of ours (la-* or inspect-*) with a running container, counted across processes."""
    out = docker("ps", "--filter", "label=com.docker.compose.project", "--format", '{{.Label "com.docker.compose.project"}}').stdout
    return len({p for p in out.split() if p.startswith(OUR_PREFIXES)})


def foreign_containers() -> list[tuple]:
    """(id, name, state) of every container not named la-* or inspect-*: John's services. Compare before and after anything we run."""
    out = docker("ps", "-a", "--format", "{{.ID}}\t{{.Names}}\t{{.State}}").stdout
    return sorted(tuple(line.split("\t")) for line in out.splitlines() if line and not line.split("\t")[1].startswith(OUR_PREFIXES))


class Registry:
    """Write-ahead record of the compose projects we start, so a crashed run's husks are known exactly."""
    def __init__(self, path: Path = REGISTRY):
        self.path = path

    def record(self, project: str, **extra):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(dict(project=project, pid=os.getpid(), t=time.time(), **extra)) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def rows(self) -> list[dict]:
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()] if self.path.exists() else []

    def names(self) -> list[str]:
        return list(dict.fromkeys(r["project"] for r in self.rows()))

    def forget(self, project: str):
        if self.path.exists():
            keep = [line for line in self.path.read_text().splitlines() if line.strip() and json.loads(line)["project"] != project]
            self.path.write_text("".join(line + "\n" for line in keep))

    def stale(self) -> dict[str, dict]:
        """Recorded projects whose recording process is dead and that still have containers, networks or volumes: {name: {containers, networks, volumes}}."""
        found = {}
        for name in self.names():
            if any(_alive(r["pid"]) for r in self.rows() if r["project"] == name):
                continue
            objs = project_objects(name)
            if any(objs.values()):
                found[name] = objs
        return found


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def project_objects(project: str) -> dict[str, list[str]]:
    flt = f"label=com.docker.compose.project={project}"
    return dict(containers=docker("ps", "-aq", "--filter", flt).stdout.split(), networks=docker("network", "ls", "-q", "--filter", flt).stdout.split(),
                volumes=docker("volume", "ls", "-q", "--filter", flt).stdout.split())


def remove_project(project: str, registry: Registry | None = None) -> dict:
    """Remove the objects labelled with exactly this compose project. Refuses names that are not la-*/inspect-* or not recorded in the registry."""
    registry = registry or Registry()
    if not project.startswith(OUR_PREFIXES) or (project.startswith("inspect-") and project not in registry.names()):
        raise ValueError(f"refusing to remove {project!r}: not ours")
    objs = project_objects(project)
    if objs["containers"]:
        docker("rm", "-f", "-v", *objs["containers"], timeout=120)
    if objs["networks"]:
        docker("network", "rm", *objs["networks"], timeout=60)
    if objs["volumes"]:
        docker("volume", "rm", "-f", *objs["volumes"], timeout=60)
    left = project_objects(project)
    if not any(left.values()):
        registry.forget(project)
    return dict(removed=objs, left=left)


class ComposeSandbox:
    """Async subset of Inspect's SandboxEnvironment (exec, write_file, read_file) over `docker compose -p <project> exec`, for the service `default`."""
    def __init__(self, project: str, compose_file: Path | str):
        self.project, self.compose_file = project, str(compose_file)

    def _base(self):
        return ["docker", "compose", "-p", self.project, "-f", self.compose_file]

    async def _run(self, argv, input: bytes | None, timeout: float):
        proc = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE if input is not None else asyncio.subprocess.DEVNULL,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(input), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(f"docker compose exec exceeded {timeout}s on the host side")
        return proc.returncode, out, err

    async def exec(self, cmd, input=None, cwd=None, env=None, user=None, timeout=None, timeout_retry=True, **_):
        from inspect_ai.util import ExecResult
        argv = self._base() + ["exec", "-T"] + (["-w", cwd] if cwd else []) + (["-u", user] if user else []) + [x for k, v in (env or {}).items() for x in ("-e", f"{k}={v}")] + ["default"]
        argv += (["timeout", "-k", "2", str(timeout)] if timeout else []) + list(cmd)
        rc, out, err = await self._run(argv, input.encode() if isinstance(input, str) else input, (timeout or 3600) + 15)
        if timeout and rc == 124:
            raise TimeoutError(f"command exceeded {timeout}s")
        return ExecResult(rc == 0, rc, out.decode(errors="replace"), err.decode(errors="replace"))

    async def write_file(self, file, contents):
        data = contents.encode() if isinstance(contents, str) else contents
        rc, _, err = await self._run(self._base() + ["exec", "-T", "default", "bash", "-c", 'mkdir -p "$(dirname "$1")" && cat > "$1"', "_", file], data, 60)
        if rc:
            raise RuntimeError(f"write_file {file}: {err.decode(errors='replace')}")

    async def read_file(self, file, text=True):
        rc, out, err = await self._run(self._base() + ["exec", "-T", "default", "cat", file], None, 120)
        if rc:
            raise FileNotFoundError(f"{file}: {err.decode(errors='replace')}")
        return out.decode(errors="replace") if text else out


class Stack:
    """One compose project of the vendored env: up(), ready(), down() by exact name. Project names should start with la-."""
    def __init__(self, env: str, project: str, registry: Registry | None = None):
        self.env_dir = ARENA / "envs" / env
        self.project, self.registry = project, registry or Registry()
        self.compose_file = self.env_dir / "codebase" / "compose.yml"
        self.sbx = ComposeSandbox(project, self.compose_file)

    async def up(self, timeout: float = 300):
        self.registry.record(self.project, compose=str(self.compose_file))
        proc = await asyncio.create_subprocess_exec(*self.sbx._base(), "up", "-d", "--quiet-pull", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise
        if proc.returncode:
            raise RuntimeError(f"docker compose up failed for {self.project}: {out.decode(errors='replace')[-800:]}")

    async def wait_ready(self, timeout: float = 240) -> float:
        """Poll envs/<env>/ready.sh in the sandbox as the harness does; returns seconds, raises RuntimeError('ready.sh did not succeed ...')."""
        t0 = time.monotonic()
        script = (self.env_dir / "ready.sh").read_text()
        while time.monotonic() - t0 < timeout:
            try:
                if (await self.sbx.exec(["bash", "-c", script], timeout=30)).success:
                    return time.monotonic() - t0
            except TimeoutError:
                pass
            await asyncio.sleep(3)
        raise RuntimeError(f"{self.env_dir.name}: ready.sh did not succeed within {timeout}s")

    async def run_script(self, name: str, timeout: float):
        path = self.env_dir / name
        await self.sbx.write_file(f"/tmp/arena_{name}", path.read_text())
        return await self.sbx.exec(["bash", f"/tmp/arena_{name}"], timeout=timeout)

    def down(self) -> dict:
        return remove_project(self.project, self.registry)


# ---- checks ----

@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str = ""
    seconds: float = 0.0
    data: dict = field(default_factory=dict)


def check_daemon(limit: float = 5.0) -> Check:
    t0 = time.monotonic()
    try:
        r = docker("version", "--format", "{{.Server.Version}}", timeout=limit)
    except DockerTimeout as e:
        return Check("docker_daemon", "fail", str(e), time.monotonic() - t0)
    dt = time.monotonic() - t0
    if r.returncode:
        return Check("docker_daemon", "fail", r.stderr.strip()[-300:], dt)
    return Check("docker_daemon", "ok" if dt < limit else "fail", f"server {r.stdout.strip()} answered in {dt:.2f}s", dt)


def image_id(env: str) -> str | None:
    r = docker("image", "inspect", "--format", "{{.Id}}", f"linuxarena/env-{env}:latest")
    return r.stdout.strip() if r.returncode == 0 else None


def check_image(env: str, lock: Path | None = None) -> Check:
    iid = image_id(env)
    if not iid:
        return Check("image", "fail", f"linuxarena/env-{env}:latest missing; run envs/{env}/build.sh")
    lock = lock or RL_DATA / "image.lock"
    if lock.exists():
        want = json.loads(lock.read_text()).get(env)
        if want and want != iid:
            return Check("image", "fail", f"image id {iid} != image.lock {want}", data=dict(image_id=iid, locked=want))
    return Check("image", "ok", iid, data=dict(image_id=iid))


def write_lock(env: str, lock: Path | None = None) -> Path:
    lock = lock or RL_DATA / "image.lock"
    data = json.loads(lock.read_text()) if lock.exists() else {}
    data[env] = image_id(env)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(data, indent=1) + "\n")
    return lock


def check_disk(path: str = "/var/lib/docker") -> Check:
    from .classify import HaltRules
    level, msg = HaltRules.disk(path)
    return Check("disk", {"ok": "ok", "warn": "warn", "halt": "fail"}[level], msg or "at least 20 GB free")


def check_memory(extra_stacks: int = 0) -> Check:
    avail, live = mem_available_gib(), live_stacks()
    need = BASE_MEM_GIB + PER_STACK_GIB * (live + extra_stacks)
    return Check("memory", "ok" if avail >= need else "fail", f"MemAvailable {avail:.1f} GiB, need {need:.1f} ({live} live + {extra_stacks} planned stacks)", data=dict(avail_gib=avail, need_gib=need, live=live))


def network_pool_size() -> int:
    """Subnets docker can hand out: from default-address-pools in /etc/docker/daemon.json when present, else the conservative 31 (a docker default of /16 blocks)."""
    try:
        pools = json.loads(Path("/etc/docker/daemon.json").read_text()).get("default-address-pools")
    except (OSError, ValueError):
        pools = None
    if not pools:
        return NET_POOL_DEFAULT
    return sum(2 ** (p["size"] - int(p["base"].split("/")[1])) for p in pools)


def check_networks(planned: int = 1) -> Check:
    r = docker("network", "ls", "-q")
    used, pool = len(r.stdout.split()), network_pool_size()
    free = pool - used
    status = "fail" if free < planned else ("warn" if free < NET_FREE_WARN else "ok")
    return Check("network_pool", status, f"{used} networks, pool {pool}, {free} free (each compose sample creates one)", data=dict(used=used, pool=pool, free=free))


def check_stale(cleanup: bool, registry: Registry | None = None) -> Check:
    registry = registry or Registry()
    stale = registry.stale()
    known = set(registry.names())
    out = docker("ps", "-a", "--format", '{{.Names}}\t{{.Label "com.docker.compose.project"}}').stdout
    unrecorded = sorted(n for n, proj in (l.split("\t") for l in out.splitlines()) if n.startswith("la-") and proj not in known)
    note = f"; unrecorded la-* containers (not touched): {unrecorded}" if unrecorded else ""
    if not stale:
        return Check("stale_projects", "ok", "no dead run left objects" + note)
    if cleanup:
        left = [n for n in stale if any(remove_project(n, registry)["left"].values())]
        return Check("stale_projects", "fail" if left else "ok", f"removed {sorted(stale)}; still present: {left}" + note)
    return Check("stale_projects", "fail", f"stale recorded projects {sorted(stale)}; rerun with --cleanup (removes exactly these)" + note)


# ---- golden control ----

PATCH_RE = re.compile(r"patch -p1 << '(\w+)'\n(.*?)\n\1\n", re.S)


def setup_patch(setup_sh: str) -> str | None:
    m = PATCH_RE.search(setup_sh)
    return m.group(2) + "\n" if m else None


async def plain_score(stack: Stack, task: Path, timeout: int = 300, timings: dict | None = None) -> tuple[bool, str]:
    """What run.py's task_scorer does for a tests.py task: restart.sh, then pytest /tmp/tests.py -x -v --tb=short in the sandbox. timings gets restart_s and tests_s."""
    t0 = time.monotonic()
    restart = await stack.run_script("restart.sh", timeout)
    if timings is not None:
        timings["restart_s"] = round(time.monotonic() - t0, 1)
    if not restart.success:
        return False, f"restart.sh failed: {(restart.stderr or restart.stdout)[-500:]}"
    await stack.sbx.write_file("/tmp/tests.py", (task / "tests.py").read_text())
    t1 = time.monotonic()
    try:
        r = await stack.sbx.exec(["pytest", "/tmp/tests.py", "-x", "-v", "--tb=short"], timeout=timeout)
    except TimeoutError:
        return False, "tests timed out"
    finally:
        if timings is not None:
            timings["tests_s"] = round(time.monotonic() - t1, 1)
    return r.returncode == 0, (r.stdout + r.stderr)[-1500:]


async def golden_task(stack: Stack, task: Path, guarded: bool = True) -> list[Check]:
    """Bug applied -> tests fail; reverse patch -> tests pass; for both the plain scorer and (when guarded) rl.guard. Leaves the tree pristine."""
    setup, out = (task / "setup.sh").read_text(), []
    patch = setup_patch(setup)
    name = f"golden:{task.name}"
    if patch is None:
        return [Check(name, "fail", "setup.sh has no 'patch -p1 << EOF' block to reverse")]
    r = await stack.sbx.exec(["bash", "-c", setup], timeout=120)
    if not r.success:
        return [Check(name, "fail", f"setup.sh failed: {r.stderr[-300:]}")]
    stages = [("bug", False), ("fixed", True)]
    for stage, want in stages:
        if stage == "fixed":
            await stack.sbx.write_file("/tmp/la_golden.patch", patch)
            r = await stack.sbx.exec(["bash", "-c", "cd /app/src && patch -R -p1 < /tmp/la_golden.patch"], timeout=60)
            if not r.success:
                return out + [Check(name, "fail", f"reverse patch failed: {(r.stdout + r.stderr)[-300:]}")]
        t0 = time.monotonic()
        ok, why = await plain_score(stack, task)
        out.append(Check(f"{name}:{stage}:plain", "ok" if ok == want else "fail", f"scorer passed={ok}, expected {want}" + ("" if ok == want else f"\n{why[-600:]}"), time.monotonic() - t0))
        if guarded:
            from . import guard
            t0 = time.monotonic()
            g = await guard.guarded_check(stack.sbx, stack.env_dir, task, guard.GuardCfg(), guard.load_reference(stack.env_dir.name))
            good = g.passed == want and (not want or not g.flags)
            out.append(Check(f"{name}:{stage}:guarded", "ok" if good else "fail", f"guarded passed={g.passed} reason={g.reason} flags={g.flags} golden_match={g.golden_match}", time.monotonic() - t0, asdict(g) | dict(final_diff=g.final_diff[:2000])))
    return out


def answer_check(task: Path) -> Check:
    accepted = [line.strip().lower() for line in (task / "answer.txt").read_text().splitlines() if line.strip()]
    ok = bool(accepted) and "zzz-wrong" not in accepted
    return Check(f"answer:{task.name}", "ok" if ok else "fail", f"{len(accepted)} accepted answers; scoring is host-side string equality in run.py (this check duplicates its logic)")


async def score_py_wrong_check(stack: Stack, task: Path) -> Check:
    """The feature scorer must reject the pristine tree (no implementation): passed False and no exception."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"doctor_score_{task.name}", task / "score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    t0 = time.monotonic()
    try:
        passed, why = await module.score(stack.sbx, "")
    except Exception as e:
        return Check(f"score_py:{task.name}", "fail", f"scorer raised {type(e).__name__}: {e}", time.monotonic() - t0)
    return Check(f"score_py:{task.name}", "ok" if not passed else "fail", f"pristine tree: passed={passed} ({str(why)[:160]!r})", time.monotonic() - t0)


async def golden_controls(env: str, guarded: bool = True, project: str | None = None) -> list[Check]:
    tasks = sorted(p for p in (ARENA / "tasks" / env).iterdir() if p.is_dir())
    stack = Stack(env, project or f"la-doctor-{os.getpid()}")
    checks = []
    t0 = time.monotonic()
    try:
        await stack.up()
        ready = await stack.wait_ready()
        checks.append(Check("doctor_stack_boot", "ok", f"{stack.project} ready in {ready:.1f}s", time.monotonic() - t0))
        for task in tasks:
            if (task / "answer.txt").exists():
                checks.append(answer_check(task))
            elif (task / "score.py").exists():
                checks.append(await score_py_wrong_check(stack, task))
        for task in tasks:
            if (task / "setup.sh").exists() and (task / "tests.py").exists():
                checks += await golden_task(stack, task, guarded)
    except Exception as e:
        checks.append(Check("golden_controls", "fail", f"{type(e).__name__}: {e}"))
    finally:
        left = stack.down()["left"]
        checks.append(Check("doctor_teardown", "ok" if not any(left.values()) else "fail", f"{stack.project} removed; left {left}"))
    return checks


def preflight(env: str = "tiktok", golden: bool = True, guarded: bool = True, cleanup: bool = False) -> list[Check]:
    """Run every check; a failed daemon check stops the rest."""
    before = foreign_containers()
    checks = [check_daemon()]
    if checks[0].status == "fail":
        return checks
    checks += [check_image(env), check_disk(), check_memory(extra_stacks=1 if golden else 0), check_networks(), check_stale(cleanup)]
    if golden and all(c.status != "fail" for c in checks):
        checks += asyncio.run(golden_controls(env, guarded))
    after = foreign_containers()
    checks.append(Check("foreign_containers", "ok" if before == after else "fail", f"non-la containers {'unchanged' if before == after else 'CHANGED'} ({len(before)} before, {len(after)} after)"))
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env", default="tiktok")
    ap.add_argument("--no-golden", action="store_true")
    ap.add_argument("--no-guard", action="store_true", help="golden controls through the plain scorer only")
    ap.add_argument("--cleanup", action="store_true", help="remove stale projects recorded in the registry, by exact name")
    ap.add_argument("--write-lock", action="store_true", help="record the current image id in rl_data/image.lock")
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.write_lock:
        print("wrote", write_lock(args.env))
    checks = preflight(args.env, not args.no_golden, not args.no_guard, args.cleanup)
    for c in checks:
        print(f"[{c.status.upper():4}] {c.name}: {c.detail} ({c.seconds:.1f}s)")
    if args.json:
        Path(args.json).write_text(json.dumps([asdict(c) for c in checks], indent=1))
    sys.exit(1 if any(c.status == "fail" for c in checks) else 0)


if __name__ == "__main__":
    main()
