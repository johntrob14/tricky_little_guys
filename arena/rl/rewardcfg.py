"""Composite-reward config. Defaults make the reward identical to the main-only episode_reward; side/CoT/final terms are OFF unless a config turns them on.

Port of the copy-.env organism reward (self-oversight/obstacles_rl.py): R = w_main*main_pass + w_side*side_done - w_final*final_reveal, with a CoT-cleanliness
penalty beta_cot gated behind cot_penalty_enabled. This module only holds weights and flags; the math lives in rl.rewards.composite_reward."""

from dataclasses import dataclass


@dataclass
class RewardCfg:
    w_main: float = 1.0
    w_side: float = 0.0
    w_final: float = 0.0
    beta_cot: float = 0.0
    side_enabled: bool = False
    cot_penalty_enabled: bool = False

    def is_main_only(self) -> bool:
        """True when every side/CoT/final term is off, so composite_reward equals episode_reward exactly."""
        return not self.side_enabled and not self.cot_penalty_enabled and self.w_side == 0.0 and self.w_final == 0.0 and self.beta_cot == 0.0
