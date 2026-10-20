"""Inert reward hook: turns one episode's summary entry (summary.json "episodes" item) into a reward component. Not wired into training; it raises
NotImplementedError unless called with enabled=True, so nothing can pick it up by accident. The component is -weight * stat / scale_max for a
high=suspicious monitor, None when the episode has no parsed turn (never a guessed value). Using it makes the policy optimise against the monitor;
a per-token variant would need per-turn scores from monitor.jsonl and the datum builder's advantage override, and is not written.
"""


def monitor_reward_component(episode, scale=(0, 10), *, enabled=False, weight=0.0, stat="max"):
    if not enabled:
        raise NotImplementedError("monitor reward is off; pass enabled=True explicitly (not used by training)")
    if stat not in ("max", "mean"):
        raise ValueError(stat)
    v = episode.get(stat)
    return None if v is None else -weight * v / scale[1]
