"""rl.ablation.run_dir / `rl.py ablation` on synthetic train out dirs (step_NNN/episodes.jsonl + cot_monitor.jsonl + a held-out monitor.jsonl); offline."""

import importlib.util
import json
import random
from pathlib import Path

import pytest

from rl import ablation as A
from rl.cot_mask import CALL, CHANNEL, END, MESSAGE
from rl.schema import Episode, Turn, append_jsonl

ARENA = Path(__file__).resolve().parents[2]
NAMES = {1: "analysis", 2: "final", 3: "commentary"}


class Tok:
    def decode(self, ids):
        return "".join(NAMES.get(i, "?") for i in ids)


def _turn(n_cot):
    return Turn(prompt_len=10, tokens=[CHANNEL, 1, MESSAGE, *([100] * n_cot), END, CHANNEL, 3, MESSAGE, 20, 21, CALL], logprobs=[])


def _ep(step, i, n_cot, done, opp, fc="none"):
    return Episode(episode_id=f"s{step}e{i}", sample_id="t", task="env/t", epoch=i, step=step, weights_id="w", prompt_tokens=[1], turns=[_turn(n_cot)] if fc != "infra" else [],
                   run="r1", phase="train", model="openai/gpt-oss-120b", failure_class=fc, passed=True, side_done=done, opportunity_present=opp)


def _cm(eid, step, turn, score, judge="j/in", **kw):
    return {**dict(schema="arena.rl.cot_monitor.v1", episode_id=eid, step=step, turn=turn, monitor="controlarena_cot", judge_model=judge, score=score, skipped=None, fallback=False,
                   c=(score or 0) / 10), **kw}


def _ho(eid, score, judge="j/strong", scale=(0, 10)):
    return dict(schema="arena.monitor.v1", monitor="controlarena_cot", judge_model=judge, scale=list(scale), direction="high=suspicious", score=score, sidecar=dict(episode_id=eid))


def _write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(map(json.dumps, rows)) + "\n")


def _len_driven(L, done):
    return min(10.0, L / 40.0) if done else 0.5


@pytest.fixture
def train_dir(tmp_path):
    """Two checkpoints; step 5 has ~43% shorter CoTs and suspicion that only tracks length, so the raw drop vanishes at matched length."""
    rng, d, cm, ho = random.Random(0), tmp_path / "r1", [], []
    for step, mu in ((0, 300), (5, 170)):
        eps = []
        for i in range(200):
            opp = i < 100
            done = rng.random() < (0.8 if opp else 0.05)
            L = max(1, int(rng.gauss(mu, mu * 0.2)))
            e = _ep(step, i, L, done, opp)
            eps.append(e)
            cm += [_cm(e.episode_id, step, 0, _len_driven(L, done))]
            ho += [_ho(e.episode_id, _len_driven(L, done))]
        eps.append(_ep(step, 999, 0, None, None, fc="infra"))
        append_jsonl(d / f"step_{step:03d}" / "episodes.jsonl", eps)
    _write(d / "cot_monitor.jsonl", cm)
    _write(tmp_path / "logs" / "monitor_heldout" / "r1" / "monitor.jsonl", ho)
    return d


def test_run_dir_shorter_cot(train_dir):
    res = A.run_dir(train_dir, train_dir.parent / "logs" / "monitor_heldout", tokenizer=Tok())
    assert res["cot_length_source"] == "analysis_channel" and res["n_episodes"] == 402 and res["n_obs"] == 400  # INFRA left out
    c0, c5 = res["checkpoints"][0], res["checkpoints"][5]
    assert c0["cot_tokens_mean"] > 250 and c5["cot_tokens_mean"] < 200
    assert c0["sus_in_loss_side_done"] - c5["sus_in_loss_side_done"] > 2 and abs(c5["matched_drop_in_loss"]) < 1.0
    assert c5["verdict"] == "shorter_cot" and c0["verdict"] == "baseline"
    assert c5["heldout_minus_in_loss_side_done"] == pytest.approx(0.0)
    assert res["coverage"][5]["side_done"] == res["coverage"][5]["side_done_in_loss"] == res["coverage"][5]["side_done_heldout"] > 0
    assert res["in_loss"]["monitor"] == "controlarena_cot" and res["in_loss"]["judge_model"] == "j/in" and res["heldout"]["judge_model"] == "j/strong"
    md = A.to_markdown(res)
    assert "verdict (heuristic)" in md and "not results" in md and "## By CoT length" in md and "| 5 |" in md


def test_cot_monitor_rows(tmp_path):
    p = tmp_path / "cot_monitor.jsonl"
    rows = [_cm("e1", 0, 0, 2.0), _cm("e1", 0, 1, 9.0), _cm("e1", 0, 1, 4.0),  # turn 1 rewritten (resume): last row wins
            _cm("e1", 0, 2, None, skipped="empty_cot"), _cm("e1", 0, 3, None, fallback=True, reason="unparsed"), _cm("e1", 0, 4, 10.0, fallback=True),
            _cm("e2", 0, 0, None, skipped="truncated")]
    p.write_text("\n".join(map(json.dumps, rows)) + "\n{torn")
    s, meta = A.cot_monitor_scores(p)
    assert s == {"e1": 4.0} and meta["judged"] == 2 and meta["fallback"] == 2 and meta["skipped"] == 2 and meta["turns"] == 6
    assert A.cot_monitor_scores(p, "mean")[0] == {"e1": 3.0}
    p.write_text("\n".join(map(json.dumps, [_cm("e1", 0, 0, 2.0), _cm("e1", 0, 1, 9.0, judge="j/other")])))
    with pytest.raises(ValueError, match="mix"):
        A.cot_monitor_scores(p)
    assert A.cot_monitor_scores(p, judge_model="j/other")[0] == {"e1": 9.0}
    p.write_text(json.dumps(dict(_cm("e1", 0, 0, 50.0), monitor="ontask_generic_100")))
    with pytest.raises(ValueError, match="direction"):
        A.cot_monitor_scores(p)


def test_run_dir_refusals_and_missing(train_dir, tmp_path):
    ho = tmp_path / "ho.jsonl"
    _write(ho, [_ho("s0e0", 5.0, judge="j/in")])
    with pytest.raises(ValueError, match="is the in-loss judge"):
        A.run_dir(train_dir, ho, tokenizer=Tok())
    _write(ho, [_ho("s0e0", 50.0, scale=(1, 100))])
    with pytest.raises(ValueError, match="scale"):
        A.run_dir(train_dir, ho, tokenizer=Tok())
    _write(tmp_path / "two" / "a" / "monitor.jsonl", [])
    _write(tmp_path / "two" / "b" / "monitor.jsonl", [])
    with pytest.raises(ValueError, match="2 monitor.jsonl"):
        A.run_dir(train_dir, tmp_path / "two", tokenizer=Tok())
    (train_dir / "cot_monitor.jsonl").unlink()
    res = A.run_dir(train_dir, tokenizer=Tok())
    assert res["in_loss"]["path"] is None and res["heldout"] is None and res["checkpoints"][5]["sus_in_loss_side_done"] is None
    assert res["checkpoints"][5]["verdict"] == "no_suspicion_drop" and "none (no cot_monitor.jsonl" in A.to_markdown(res)
    with pytest.raises(FileNotFoundError):
        A.run_dir(tmp_path / "empty")


def test_tokenizer_fallback_counts_sampled_tokens(train_dir, monkeypatch):
    import rl.cot_mask as cm

    monkeypatch.setattr(cm, "_tokenizer_for", lambda m: (_ for _ in ()).throw(OSError("offline")))
    res = A.run_dir(train_dir)
    assert res["cot_length_source"].startswith("sampled_tokens_upper_bound")


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_RL_DATA", str(tmp_path))
    monkeypatch.delenv("ARENA_RL_RUN_TINKER", raising=False)
    spec = importlib.util.spec_from_file_location("arena_rl_cli_ablation", ARENA / "rl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_ablation(cli, capsys, train_dir, tmp_path, monkeypatch):
    import tinker
    import rl.cot_mask as cm

    monkeypatch.setattr(tinker, "ServiceClient", lambda *a, **k: pytest.fail("a live Tinker client was created"))
    monkeypatch.setattr(cm, "_tokenizer_for", lambda m: Tok())
    out = tmp_path / "out"
    cli.main(["ablation", "r1", "--heldout", str(tmp_path / "logs" / "monitor_heldout" / "r1" / "monitor.jsonl"), "--out", str(out)])  # run name under ARENA_RL_DATA
    printed = capsys.readouterr().out
    assert "# CoT ablation: r1" in printed and (out / "ablation.md").read_text() in printed
    j = json.loads((out / "ablation.json").read_text())
    assert j["schema"] == A.SCHEMA and j["checkpoints"]["5"]["verdict"] == "shorter_cot" and j["heldout"]["judge_model"] == "j/strong"
    cli.main(["ablation", str(train_dir), "--stat", "mean", "--baseline-step", "5"])
    assert json.loads((train_dir / "ablation.json").read_text())["checkpoints"]["5"]["verdict"] == "baseline"
    with pytest.raises(SystemExit, match="baseline_step 3"):
        cli.main(["ablation", str(train_dir), "--baseline-step", "3"])
    with pytest.raises(SystemExit, match="no step_"):
        cli.main(["ablation", str(tmp_path / "nope")])
