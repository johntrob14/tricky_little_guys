"""Hard-capped budget with worst-case reservation and an append-only csv ledger (SPEC 10). Prices are USD per million tokens at list price."""

import csv
import fcntl
import os
import threading
import time
from pathlib import Path

DATA_DIR = Path(os.environ.get("ARENA_RL_DATA", Path(__file__).resolve().parent.parent / "rl_data"))
TOTAL_CAP_USD = 20.0
COLUMNS = ["time", "phase", "run", "model", "kind", "estimate_usd", "counted_usd", "note"]
# model: (prefill, cached prefill, sample, train)
PRICES = {"openai/gpt-oss-120b": (0.33, 0.066, 0.84, 0.737), "openai/gpt-oss-20b": (0.18, 0.036, 0.45, 0.396)}


class BudgetExceeded(RuntimeError):
    pass


def _price(model: str) -> tuple[float, float, float, float]:
    return PRICES[model.split(":")[0]]


def sample_cost_usd(model: str, prompt_tokens: int, cached: int, completion: int) -> float:
    pre, hit, smp, _ = _price(model)
    return ((prompt_tokens - cached) * pre + cached * hit + completion * smp) / 1e6


def train_cost_usd(model: str, n_tokens: int) -> float:
    return n_tokens * _price(model)[3] / 1e6


def episode_worst_case_usd(model: str, *, max_turns: int, max_tokens: int, ctx: int = 32768, p0: int = 3000, growth: int | None = None) -> float:
    """Upper bound for one episode with no prefix-cache hits: each turn's prompt grows by `growth` tokens (default max_tokens + 1000) up to the context
    and samples at most max_tokens."""
    growth = growth if growth is not None else max_tokens + 1000
    total = 0.0
    for k in range(max_turns):
        p = min(ctx - 1, p0 + k * growth)
        total += sample_cost_usd(model, p, 0, min(max_tokens, ctx - p))
    return total


class Ledger:
    """Append-only csv, flock + fsync per row so concurrent processes interleave whole rows."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else DATA_DIR / "ledger.csv"

    def append(self, phase: str, run: str, model: str, kind: str, estimate_usd: float | None = None, counted_usd: float | None = None, note: str = "") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", newline="") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            if f.tell() == 0:
                csv.writer(f).writerow(COLUMNS)
            csv.writer(f).writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), phase, run, model, kind,
                                    "" if estimate_usd is None else f"{estimate_usd:.6f}", "" if counted_usd is None else f"{counted_usd:.6f}", note])
            f.flush()
            os.fsync(f.fileno())

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, newline="") as f:
            return list(csv.DictReader(f))

    def total_counted(self) -> float:
        return sum(float(r["counted_usd"]) for r in self.rows() if r["kind"] == "counted" and r["counted_usd"])


class Budget:
    """One per stage and process. `committed` is counted spend, `reserved` is the sum of live worst-case reservations. Nothing may start if
    committed + reserved + request would pass the cap (a cap is a floor, not permission)."""

    def __init__(self, cap_usd: float, model: str = "openai/gpt-oss-120b", *, stage: str = "stage", run: str = "", ledger: Ledger | None = None,
                 total_cap_usd: float = TOTAL_CAP_USD):
        self.cap, self.model, self.stage, self.run = cap_usd, model, stage, run
        self.ledger = ledger or Ledger()
        self.total_cap = total_cap_usd
        self.committed = 0.0
        self.reserved = 0.0
        self.by_kind: dict[str, float] = {}
        self.tokens = dict(prompt=0, cached=0, completion=0, train=0)
        self._lock = threading.Lock()
        self.started = False

    @property
    def remaining(self) -> float:
        return self.cap - self.committed - self.reserved

    def start_stage(self, forecast_usd: float, note: str = "") -> None:
        """Refuse to start when the forecast is over the stage cap or the ledger total plus forecast is over the total cap; else ledger a forecast row."""
        total = self.ledger.total_counted()
        if forecast_usd > self.cap:
            raise BudgetExceeded(f"stage {self.stage}: forecast ${forecast_usd:.4f} is over the stage cap ${self.cap:.2f}")
        if total + forecast_usd > self.total_cap:
            raise BudgetExceeded(f"stage {self.stage}: ledger total ${total:.4f} + forecast ${forecast_usd:.4f} is over the total cap ${self.total_cap:.2f}")
        self.ledger.append(self.stage, self.run, self.model, "forecast", estimate_usd=forecast_usd, note=note)
        self.started = True

    def check_total(self, forecast_usd: float) -> None:
        """Refuse when the ledger's counted total plus this forecast would pass the total cap."""
        total = self.ledger.total_counted()
        if total + forecast_usd > self.total_cap:
            raise BudgetExceeded(f"stage {self.stage}: ledger total ${total:.4f} + ${forecast_usd:.4f} is over the total cap ${self.total_cap:.2f}")

    def require(self, forecast_usd: float) -> None:
        with self._lock:
            if self.committed + self.reserved + forecast_usd > self.cap:
                raise BudgetExceeded(f"stage {self.stage}: ${self.committed:.4f} counted + ${self.reserved:.4f} reserved + ${forecast_usd:.4f} forecast is over the cap ${self.cap:.2f}")

    def reserve(self, worst_case_usd: float) -> None:
        with self._lock:
            if self.committed + self.reserved + worst_case_usd > self.cap:
                raise BudgetExceeded(f"stage {self.stage}: reserving ${worst_case_usd:.4f} on ${self.committed:.4f} counted + ${self.reserved:.4f} reserved passes the cap ${self.cap:.2f}")
            self.reserved += worst_case_usd

    def release(self, worst_case_usd: float) -> None:
        with self._lock:
            self.reserved = max(0.0, self.reserved - worst_case_usd)

    def commit(self, actual_usd: float, *, reserved_usd: float = 0.0, kind: str = "sample", note: str = "", model: str | None = None) -> None:
        """Replace a reservation (if any) with the counted cost, and ledger it. Counted cost over the cap is kept (it was spent) and raises. `model` is the
        row's model column, default the stage's (policy) model; judge calls pass the judge model."""
        with self._lock:
            self.reserved = max(0.0, self.reserved - reserved_usd)
            self.committed += actual_usd
            self.by_kind[kind] = self.by_kind.get(kind, 0.0) + actual_usd
            over = self.committed > self.cap + 1e-12
        self.ledger.append(self.stage, self.run, model or self.model, "counted", counted_usd=actual_usd, note=f"{kind} {note}".strip())
        if over:
            raise BudgetExceeded(f"stage {self.stage}: counted ${self.committed:.4f} passed the cap ${self.cap:.2f}")

    def summary(self) -> dict:
        return dict(stage=self.stage, cap_usd=self.cap, committed_usd=round(self.committed, 6), reserved_usd=round(self.reserved, 6), by_kind={k: round(v, 6) for k, v in self.by_kind.items()}, tokens=self.tokens)
