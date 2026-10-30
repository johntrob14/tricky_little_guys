"""Machine-wide cap on concurrent Inspect docker sandboxes.

Inspect's max_sandboxes limits one process. Two honeypot runs of 8 each put 16 tiktok stacks (about 0.6 GiB each)
on a 14 GiB box, and the kernel OOM killer then took down the user session (2026-10-21 17:58 UTC). capped() shrinks a
requested max_sandboxes by the slots other live runs have reserved, the stacks already up, and the memory left, and
exits if nothing fits. Reservations are per pid under $XDG_RUNTIME_DIR, taken under a lock so two runs started
together cannot both see an empty machine.

    ARENA_MAX_SANDBOXES=6   total stacks across all runs (default 6)
    ARENA_GIB_PER_STACK=1.0 memory budgeted per stack (measured 0.5 to 0.65 GiB resident, room for a scorer spike)
    ARENA_RESERVE_GIB=3.0   memory kept free for everything else
"""

import fcntl
import os
import subprocess
from pathlib import Path

TOTAL = int(os.environ.get("ARENA_MAX_SANDBOXES", "6"))
GIB_PER_STACK = float(os.environ.get("ARENA_GIB_PER_STACK", "1.0"))
RESERVE_GIB = float(os.environ.get("ARENA_RESERVE_GIB", "3.0"))
SLOTS = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "arena-sandbox-slots"


def running_stacks() -> int:
    """Distinct Inspect compose projects with a container up; 0 if docker is unreachable."""
    try:
        out = subprocess.run(["docker", "ps", "--format", '{{.Label "com.docker.compose.project"}}'], capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return 0
    return len({p for p in out if p.startswith("inspect-")})


def available_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    raise RuntimeError("MemAvailable not in /proc/meminfo")


def reserved_by_others() -> int:
    total = 0
    for f in SLOTS.glob("*"):
        if f.name.isdigit() and int(f.name) != os.getpid() and Path(f"/proc/{f.name}").exists():
            total += int(f.read_text() or 0)
        elif f.name.isdigit() and not Path(f"/proc/{f.name}").exists():
            f.unlink(missing_ok=True)
    return total


def capped(requested: int) -> int:
    """max_sandboxes to pass to Inspect: requested, reduced to what the machine can hold alongside other runs."""
    SLOTS.mkdir(parents=True, exist_ok=True)
    with open(SLOTS / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        taken = max(running_stacks(), reserved_by_others())
        by_memory = int((available_gib() - RESERVE_GIB) // GIB_PER_STACK)
        n = min(requested, TOTAL - taken, by_memory)
        if n < 1:
            raise SystemExit(f"sandbox_cap: no room for a sandbox (cap {TOTAL}, {taken} stacks up or reserved, {available_gib():.1f} GiB available, "
                             f"{RESERVE_GIB:.0f} GiB kept free). Stop finished runs, or `docker ps` for orphaned inspect-* stacks left by a killed run.")
        (SLOTS / str(os.getpid())).write_text(str(n))
    if n < requested:
        print(f"sandbox_cap: max_sandboxes {requested} -> {n} (cap {TOTAL}, {taken} up or reserved, {available_gib():.1f} GiB available)", flush=True)
    return n
