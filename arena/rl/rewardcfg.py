"""Composite-reward config. Defaults make the reward identical to the main-only episode_reward; side/CoT/final terms are OFF unless a config turns them on.

Port of the copy-.env organism reward (self-oversight/obstacles_rl.py): R = w_main*main_pass + w_side*side_done - w_final*final_reveal, with a CoT-cleanliness
penalty beta_cot gated behind cot_penalty_enabled. This module only holds weights and flags; the math lives in rl.rewards.composite_reward."""

import math
from dataclasses import dataclass, fields


@dataclass
class RewardCfg:
    w_main: float = 1.0
    w_side: float = 0.0
    w_final: float = 0.0
    beta_cot: float = 0.0
    side_enabled: bool = False
    cot_penalty_enabled: bool = False

    def is_main_only(self) -> bool:
        """True when every side/CoT/final term is off and w_main is 1, so composite_reward equals episode_reward exactly."""
        return self.w_main == 1.0 and not self.side_enabled and not self.cot_penalty_enabled and self.w_side == 0.0 and self.w_final == 0.0 and self.beta_cot == 0.0


WEIGHTS = ("w_main", "w_side", "w_final", "beta_cot")
FLAGS = ("side_enabled", "cot_penalty_enabled")


def reward_cfg_from_dict(raw: dict | None) -> RewardCfg:
    """RewardCfg from an organism config's reward_cfg object (missing keys keep the defaults). ValueError on an unknown key, a non-bool flag, or a
    weight that is not a finite number >= 0 (w_final is subtracted by composite_reward, so its sign is fixed there)."""
    if raw is None:
        return RewardCfg()
    if not isinstance(raw, dict):
        raise ValueError(f"reward_cfg must be an object, got {raw!r}")
    d = {k: v for k, v in raw.items() if not k.startswith("_")}
    bad = sorted(set(d) - {f.name for f in fields(RewardCfg)})
    if bad:
        raise ValueError(f"reward_cfg: unknown field(s) {bad}; allowed {sorted(WEIGHTS + FLAGS)}")
    for k in FLAGS:
        if k in d and not isinstance(d[k], bool):
            raise ValueError(f"reward_cfg.{k} must be true/false, got {d[k]!r}")
    for k in WEIGHTS:
        if k in d:
            check_weight(f"reward_cfg.{k}", d[k])
            d[k] = float(d[k])
    return RewardCfg(**d)


def check_weight(name: str, v) -> None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be a finite number >= 0, got {v!r}")
