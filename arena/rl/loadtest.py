"""Docker-only concurrency measurement of Ethan's compose stack (no Inspect, no Tinker).

    python -m rl.loadtest [--levels 1,4,6,8] [--task video_url_format_validation] [--no-guard] [--out rl_data/loadtest.json]

For each level N it brings up N stacks (compose projects la-load-<N>-<i>) at once, waits for envs/<env>/ready.sh, then runs the scoring path on every
stack at once (restart.sh + the task's tests, as run.py's scorer does, and then rl.guard's guarded_check), while sampling host MemAvailable, host CPU,
and each stack's cgroup memory.current/memory.peak and cpu usage. A level is not started when MemAvailable minus the projected stack memory would drop
below 2.5 GiB, and a running level is aborted when MemAvailable falls below 2.5 GiB. Everything is torn down by exact project name.
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

from .doctor import ARENA, RL_DATA, Registry, Stack, check_daemon, foreign_containers, mem_available_gib, plain_score, project_containers

FLOOR_GIB = 2.5
CGROUP_ROOTS = ["/sys/fs/cgroup/system.slice/docker-{id}.scope", "/sys/fs/cgroup/docker/{id}"]


def read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def cgroup_dir(cid: str) -> Path | None:
    for pat in CGROUP_ROOTS:
        p = Path(pat.format(id=cid))
        if p.exists():
            return p
    return None


def cpu_usec(cg: Path) -> int | None:
    try:
        for line in (cg / "cpu.stat").read_text().splitlines():
            if line.startswith("usage_usec"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def host_cpu() -> tuple[int, int]:
    f = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    return f[3] + f[4], sum(f)


class Sampler:
    """Polls MemAvailable, host CPU busy fraction and every watched container's cgroup once a second; sets .abort when MemAvailable < floor."""
    def __init__(self, floor_gib: float = FLOOR_GIB):
        self.floor, self.abort, self.low_gib = floor_gib, False, mem_available_gib()
        self.cpu_peak, self.containers, self.peak, self.cpu0, self.cpu_last = 0.0, {}, {}, {}, {}
        self._last = host_cpu()
        self._stop = False
        self.mem_series = []

    def watch(self, name: str, cid: str):
        cg = cgroup_dir(cid)
        if cg:
            self.containers[name] = cg
            self.cpu0[name] = cpu_usec(cg)

    def projects_peak(self, project: str) -> dict:
        mine = {n: v for n, v in self.peak.items() if n.startswith(project + "-")}
        cpu = {n: (self.cpu_last.get(n, 0) - (self.cpu0.get(n) or 0)) / 1e6 for n in mine if self.cpu_last.get(n) is not None}
        return dict(peak_bytes=sum(mine.values()), by_container={n: round(v / 1e6, 1) for n, v in mine.items()}, cpu_seconds=round(sum(cpu.values()), 1))

    async def run(self):
        while not self._stop:
            avail = mem_available_gib()
            self.low_gib = min(self.low_gib, avail)
            self.mem_series.append(round(avail, 2))
            if avail < self.floor:
                self.abort = True
            idle, total = host_cpu()
            d_idle, d_total = idle - self._last[0], total - self._last[1]
            self._last = (idle, total)
            if d_total:
                self.cpu_peak = max(self.cpu_peak, 1 - d_idle / d_total)
            for name, cg in list(self.containers.items()):
                cur = max(read_int(cg / "memory.peak") or 0, read_int(cg / "memory.current") or 0)
                self.peak[name] = max(self.peak.get(name, 0), cur)
                self.cpu_last[name] = cpu_usec(cg)
            await asyncio.sleep(1)

    def stop(self):
        self._stop = True


async def one_stack(stack: Stack, sampler: Sampler, task: Path, guard: bool, row: dict, t_start: float):
    """Boot, wait ready, restart.sh + tests (plain), then guarded_check; fills row with timings. Exceptions are recorded, not raised."""
    try:
        await stack.up()
        row["up_s"] = round(time.monotonic() - t_start, 1)
        for c in project_containers(stack.project):
            sampler.watch(c["Names"], c["ID"])
        await stack.wait_ready()
        row["boot_s"] = round(time.monotonic() - t_start, 1)
        row["ready_wait_s"] = round(row["boot_s"] - row["up_s"], 1)
    except Exception as e:
        row["error"] = f"boot: {type(e).__name__}: {str(e)[-300:]}"
        return


async def score_phase(stack: Stack, sampler: Sampler, task: Path, guard: bool, row: dict):
    t0 = time.monotonic()
    try:
        timings = {}
        ok, why = await plain_score(stack, task, timings=timings)
        row.update(timings)
        row["plain_score_s"] = round(time.monotonic() - t0, 1)
        row["tests_passed"] = ok
        if guard:
            from . import guard as g
            t2 = time.monotonic()
            res = await g.guarded_check(stack.sbx, stack.env_dir, task, g.GuardCfg(), g.load_reference(stack.env_dir.name))
            row["guarded_score_s"] = round(time.monotonic() - t2, 1)
            row["guarded_passed"], row["guarded_reason"], row["guarded_flags"], row["guarded_steps"] = res.passed, res.reason, res.flags, res.steps
    except Exception as e:
        row["error"] = f"score: {type(e).__name__}: {str(e)[-300:]}"


async def run_level(env: str, n: int, task: Path, guard: bool, registry: Registry, floor: float) -> dict:
    stacks = [Stack(env, f"la-load-{n}-{i}", registry) for i in range(n)]
    sampler = Sampler(floor)
    rows = [dict(project=s.project) for s in stacks]
    avail0 = mem_available_gib()
    level = dict(n=n, mem_available_start_gib=round(avail0, 2), networks_before=len(__import__("subprocess").run(["docker", "network", "ls", "-q"], capture_output=True, text=True).stdout.split()))
    sampler_task = asyncio.create_task(sampler.run())
    t0 = time.monotonic()
    try:
        await asyncio.gather(*(one_stack(s, sampler, task, guard, r, t0) for s, r in zip(stacks, rows)))
        level["boot_wall_s"] = round(time.monotonic() - t0, 1)
        level["networks_at_peak"] = len(__import__("subprocess").run(["docker", "network", "ls", "-q"], capture_output=True, text=True).stdout.split())
        level["mem_after_boot_gib"] = round(mem_available_gib(), 2)
        if sampler.abort:
            level["aborted"] = f"MemAvailable fell below {floor} GiB during boot (low {sampler.low_gib:.2f})"
        else:
            t1 = time.monotonic()
            await asyncio.gather(*(score_phase(s, sampler, task, guard, r) for s, r in zip(stacks, rows) if "error" not in r))
            level["score_wall_s"] = round(time.monotonic() - t1, 1)
            if sampler.abort:
                level["aborted"] = f"MemAvailable fell below {floor} GiB during scoring (low {sampler.low_gib:.2f})"
    finally:
        sampler.stop()
        await sampler_task
        for s, r in zip(stacks, rows):
            r.update(sampler.projects_peak(s.project))
            r["peak_rss_gib"] = round(r.pop("peak_bytes") / 1024 ** 3, 3)
            left = s.down()["left"]
            r["teardown_clean"] = not any(left.values())
    level["mem_available_low_gib"] = round(sampler.low_gib, 2)
    level["host_cpu_peak_frac"] = round(sampler.cpu_peak, 3)
    level["mem_series_gib"] = sampler.mem_series[::5]
    level["stacks"] = rows
    peaks = [r["peak_rss_gib"] for r in rows if r.get("peak_rss_gib")]
    if peaks:
        level["peak_rss_gib_per_stack"] = dict(max=max(peaks), mean=round(statistics.mean(peaks), 3))
    for key in ("boot_s", "restart_s", "tests_s", "plain_score_s", "guarded_score_s"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            level[key] = dict(median=round(statistics.median(vals), 1), max=max(vals))
    return level


async def main_async(args) -> dict:
    env, task = args.env, ARENA / "tasks" / args.env / args.task
    registry = Registry()
    before = foreign_containers()
    out = dict(env=env, task=args.task, host=dict(cores=os.cpu_count(), mem_available_gib_start=round(mem_available_gib(), 2)), floor_gib=args.floor, levels=[])
    out["foreign_before"] = before
    per_stack = 0.9
    for n in args.levels:
        avail = mem_available_gib()
        if avail - n * per_stack < args.floor:
            out["levels"].append(dict(n=n, skipped=f"admission: MemAvailable {avail:.1f} GiB - {n} x {per_stack:.2f} GiB projected < {args.floor} GiB floor"))
            continue
        level = await run_level(env, n, task, not args.no_guard, registry, args.floor)
        out["levels"].append(level)
        if level.get("peak_rss_gib_per_stack"):
            per_stack = max(per_stack, level["peak_rss_gib_per_stack"]["max"])
        if level.get("aborted"):
            break
    after = foreign_containers()
    out["foreign_after"] = after
    out["foreign_unchanged"] = before == after
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env", default="tiktok")
    ap.add_argument("--task", default="video_url_format_validation", help="a tests.py task; its tests run on the pristine tree (they pass)")
    ap.add_argument("--levels", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4, 6, 8])
    ap.add_argument("--floor", type=float, default=FLOOR_GIB)
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--out", default=str(RL_DATA / "loadtest.json"))
    args = ap.parse_args()
    d = check_daemon()
    if d.status == "fail":
        raise SystemExit(f"docker: {d.detail}")
    result = asyncio.run(main_async(args))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    for lv in result["levels"]:
        print({k: v for k, v in lv.items() if k not in ("stacks", "mem_series_gib")})
    print("foreign containers unchanged:", result["foreign_unchanged"])


if __name__ == "__main__":
    main()
