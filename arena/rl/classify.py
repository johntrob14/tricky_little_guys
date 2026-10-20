"""Failure classification for RL episodes (SPEC_AMENDMENTS G1-G4): every failed episode is AGENT, INFRA or HALT.

AGENT: the model's doing; reward 0, counts in pass rates. INFRA: docker, host or Tinker misbehaved; reward None, the episode is rerun from scratch in a
new sandbox (max 2 reruns), its tokens are dropped from training and its cost ledgered as wasted. HALT: stop launching work and write HALTED.md.

    outcome = await run_with_retries(lambda attempt: run_episode(attempt), RetryPolicy(), Ledger(), HaltRules())

Inspect route: pass **INSPECT_RETRY to eval() (retry_on_error reruns a failed sample from scratch in a fresh sandbox) and classify_log(log) afterwards.
"""

import asyncio
import re
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

AGENT, INFRA, HALT = "AGENT", "INFRA", "HALT"
MAX_INFRA_RETRIES = 2
BACKOFF = (5, 15, 45)
INSPECT_RETRY = dict(retry_on_error=MAX_INFRA_RETRIES)
DISK_WARN_GB, DISK_HALT_GB = 20, 10


@dataclass
class Failure:
    failure_class: str
    reason: str
    detail: str = ""

    def row(self) -> dict:
        return dict(failure_class=self.failure_class, failure_reason=self.reason)


# (regex on lowercased text, class, reason); first match wins, so specific rules come first.
RULES = [
    (r"budgetexceeded|budget exceeded", HALT, "budget_exceeded"),
    (r"authenticationerror|unauthorized|invalid api key|permission_denied|\b40[13]\b.*tinker|tinker.*\b40[13]\b", HALT, "tinker_auth"),
    (r"cannot connect to the docker daemon|is the docker daemon running|error during connect|dial unix .*docker\.sock|docker daemon (is )?unreachable", INFRA, "docker_daemon_unreachable"),
    (r"all predefined address pools have been fully subnetted|could not find an available, non-overlapping ipv4 address pool", INFRA, "network_pool_exhausted"),
    (r"no space left on device", INFRA, "disk_full"),
    (r"container name .* is already in use|conflict\. the container name|network .* already exists", INFRA, "name_conflict"),
    (r"pull access denied|manifest unknown|no such image|image .* not found|repository does not exist", INFRA, "image_missing"),
    (r"toomanyrequests|rate limit.*docker|you have reached your (pull|unauthenticated)", INFRA, "registry_rate_limited"),
    (r"port is already allocated|address already in use", INFRA, "port_conflict"),
    (r"oomkilled|out of memory|exit(ed)? (with )?code 137|returned non-zero exit status 137|killed process", INFRA, "oom_killed"),
    (r"ready\.sh did not succeed", INFRA, "ready_timeout"),
    (r"setup\.sh failed", INFRA, "setup_failed"),
    (r"app not ready \d+s after restart", AGENT, "app_not_ready"),
    (r"tests timed out after", AGENT, "tests_timeout"),
    (r"is not running|no such container|container .* (not found|has exited|is gone)|service \"default\" is not running|container_lost|cannot exec in a stopped container|is paused", INFRA, "container_lost"),
    (r"docker compose .*(up|start).*(fail|error|exit)|compose up|failed to (start|create) |error response from daemon|dependency failed to start|unhealthy", INFRA, "compose_up_failed"),
    (r"restart\.sh failed", INFRA, "restart_failed"),
    (r"tinker|apiconnectionerror|apitimeouterror|ratelimiterror|internalservererror|service unavailable|\b(429|502|503|504)\b|sampling client|connection (reset|refused|aborted)", INFRA, "tinker_error"),
    (r"timeouterror|timed out|timeout", INFRA, "timeout"),
]


def classify_text(text: str, phase: str | None = None, container_running: bool | str | None = None) -> Failure:
    """Classify an error message. phase in {create, ready, setup, episode, exec, score, tinker}; container_running, when known (bool, or the state string from doctor.container_state),
    decides whether an exec timeout is the agent's slow command (running) or a hung/lost container (paused, exited, gone, oomkilled)."""
    low = (text or "").lower()
    state = {True: "running", False: "gone"}.get(container_running, container_running)
    if phase == "exec" and re.search(r"timeouterror|timed out|timeout", low):
        if state == "running":
            return Failure(AGENT, "tool_timeout", text[-500:])
        return Failure(INFRA, "container_lost" if state in ("gone", "exited") else "oom_killed" if state == "oomkilled" else "exec_hang", text[-500:])
    if state == "oomkilled":
        return Failure(INFRA, "oom_killed", text[-500:])
    if phase == "score" and re.search(r"timeouterror|timed out|timeout", low) and not re.search(r"tests timed out after", low):
        return Failure(INFRA, "scorer_timeout", text[-500:])
    if phase == "tinker":
        return Failure(HALT, "tinker_auth", text[-500:]) if re.search(r"unauthorized|authentication|invalid api key|\b40[13]\b", low) else Failure(INFRA, "tinker_error", text[-500:])
    for pattern, cls, reason in RULES:
        if re.search(pattern, low):
            return Failure(cls, reason, text[-500:])
    return Failure(INFRA, "unclassified", text[-500:])


def classify_exception(exc: BaseException, phase: str | None = None, container_running: bool | None = None) -> Failure:
    if isinstance(exc, EpisodeFailure):
        return exc.failure
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) and phase in ("exec", "score"):
        return classify_text(f"TimeoutError {exc}", phase, container_running)
    return classify_text(f"{type(exc).__name__}: {exc}", phase, container_running)


def classify_score(explanation: str) -> Failure | None:
    """A scored sample whose explanation shows a scorer-side problem; None for an ordinary pass/fail."""
    low = (explanation or "").lower()
    if "restart.sh failed" in low:
        if "app not ready" in low:
            return Failure(AGENT, "app_not_ready", explanation[:500])
        found = classify_text(explanation)
        return found if found.reason not in ("unclassified", "timeout") else Failure(INFRA, "restart_failed", explanation[:500])
    if "tests timed out after" in low:
        return Failure(AGENT, "tests_timeout", explanation[:500])
    return None


def classify_sample(sample) -> Failure | None:
    """Failure of an Inspect EvalSample, or None when it ran and scored cleanly (a wrong answer is not a failure)."""
    if sample.error is not None:
        return classify_text(f"{sample.error.message}\n{sample.error.traceback or ''}", None)
    if sample.limit is not None:
        return Failure(AGENT, f"limit_{sample.limit.type}", str(sample.limit.limit))
    for score in (sample.scores or {}).values():
        failure = classify_score(score.explanation or "")
        if failure:
            return failure
    return None


def classify_log(log) -> list[dict]:
    """One row per sample of an EvalLog: id, epoch, failure_class (None, AGENT or INFRA), failure_reason, attempt (1 + retries Inspect made), retry_reasons."""
    rows = []
    for s in log.samples or []:
        failure = classify_sample(s)
        retries = [classify_text(f"{e.message}\n{e.traceback or ''}").reason for e in (s.error_retries or [])]
        rows.append(dict(id=s.id, epoch=s.epoch, failure_class=failure.failure_class if failure else None, failure_reason=failure.reason if failure else None,
                         attempt=1 + len(retries), retry_reasons=retries))
    return rows


class EpisodeFailure(Exception):
    """Raise from an episode to force a class; cost_usd is what the failed attempt spent (ledgered as wasted if INFRA)."""
    def __init__(self, failure: Failure, cost_usd: float = 0.0):
        super().__init__(f"{failure.failure_class}:{failure.reason} {failure.detail[:200]}")
        self.failure, self.cost_usd = failure, cost_usd


class Halt(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class RetryPolicy:
    max_infra_retries: int = MAX_INFRA_RETRIES
    backoff: tuple = BACKOFF

    def should_retry(self, failure: Failure, attempt: int) -> bool:
        """attempt is 1-based, the attempt that just failed."""
        return failure.failure_class == INFRA and attempt <= self.max_infra_retries


@dataclass
class Ledger:
    """Cost accounting for lost attempts. on_wasted(usd, failure) is the hook for the budget tracker."""
    wasted_usd: float = 0.0
    attempts: list = field(default_factory=list)
    on_wasted: object = None

    def add(self, attempt: int, failure: Failure, cost_usd: float = 0.0):
        wasted = cost_usd if failure.failure_class == INFRA else 0.0
        self.wasted_usd += wasted
        self.attempts.append(dict(attempt=attempt, cost_usd=cost_usd, wasted_usd=wasted, **failure.row()))
        if wasted and self.on_wasted:
            self.on_wasted(wasted, failure)


@dataclass
class HaltRules:
    """G4: 3 consecutive INFRA attempts; INFRA rate above 25 percent over the last 8 attempts (full window); docker unreachable for over 10 min; disk gate."""
    consecutive: int = 3
    window: int = 8
    max_rate: float = 0.25
    docker_down_s: float = 600.0
    recent: deque = field(default_factory=lambda: deque(maxlen=8))
    run: int = 0
    docker_down_since: float | None = None

    def record(self, failure: Failure | None) -> str | None:
        """Record one finished attempt (None = it ended AGENT-side or clean). Returns a halt reason or None."""
        if failure is not None and failure.failure_class == HALT:
            return f"halt_class:{failure.reason}"
        infra = failure is not None and failure.failure_class == INFRA
        self.run = self.run + 1 if infra else 0
        self.recent.append(infra)
        if self.run >= self.consecutive:
            return f"{self.run}_consecutive_infra_failures"
        if len(self.recent) == self.window and sum(self.recent) / self.window > self.max_rate:
            return f"infra_rate_{sum(self.recent)}_of_{self.window}"
        return None

    def note_docker(self, reachable: bool, now: float | None = None) -> str | None:
        now = time.monotonic() if now is None else now
        if reachable:
            self.docker_down_since = None
            return None
        self.docker_down_since = self.docker_down_since if self.docker_down_since is not None else now
        return "docker_unreachable_over_10min" if now - self.docker_down_since > self.docker_down_s else None

    @staticmethod
    def disk(path: str | Path = "/var/lib/docker") -> tuple[str, str | None]:
        """('ok'|'warn'|'halt', message) from free space on the filesystem holding path (falls back to /)."""
        p = Path(path) if Path(path).exists() else Path("/")
        free = shutil.disk_usage(p).free / 1e9
        if free < DISK_HALT_GB:
            return "halt", f"disk_low_{free:.1f}GB_free"
        return ("warn", f"disk_{free:.1f}GB_free") if free < DISK_WARN_GB else ("ok", None)


def write_halted(run_dir: str | Path, reason: str, detail: str = "") -> Path:
    path = Path(run_dir) / "HALTED.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# HALTED\n\nReason: {reason}\n\n{detail}\n\nNo further work was launched. Resume with run_many(resume=True) after fixing the cause.\n")
    return path


@dataclass
class Outcome:
    result: object
    failure: Failure | None
    attempt: int
    failures: list = field(default_factory=list)

    @property
    def reward_valid(self) -> bool:
        """False when the episode ended INFRA: reward None, exclude from group means and pass-rate denominators."""
        return not (self.failure and self.failure.failure_class == INFRA)

    def row(self) -> dict:
        return dict(attempt=self.attempt, **(self.failure.row() if self.failure else dict(failure_class=None, failure_reason=None)), infra_history=[asdict(f) for f in self.failures])


async def run_with_retries(fn, policy: RetryPolicy | None = None, ledger: Ledger | None = None, halt: HaltRules | None = None, phase: str | None = None, sleep=asyncio.sleep) -> Outcome:
    """Run `await fn(attempt)` (attempt 1-based; each call must build a new sandbox). Exceptions are classified: AGENT ends the episode with the failure,
    INFRA is retried up to policy.max_infra_retries times with backoff (the lost attempt is ledgered), HALT and halt-rule hits raise Halt."""
    policy, ledger = policy or RetryPolicy(), ledger if ledger is not None else Ledger()
    failures = []
    attempt = 0
    while True:
        attempt += 1
        try:
            result = await fn(attempt)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure = classify_exception(exc, phase)
            ledger.add(attempt, failure, getattr(exc, "cost_usd", 0.0))
            reason = halt.record(failure) if halt else None
            if failure.failure_class == HALT or reason:
                raise Halt(reason or f"{failure.failure_class}:{failure.reason}") from exc
            if policy.should_retry(failure, attempt):
                failures.append(failure)
                await sleep(policy.backoff[min(attempt - 1, len(policy.backoff) - 1)])
                continue
            return Outcome(None, failure, attempt, failures)
        if halt:
            halt.record(None)
        return Outcome(result, None, attempt, failures)
