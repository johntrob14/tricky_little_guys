"""rl.report and the report / plan-collect CLI on synthetic episode files: no Tinker, no docker, no network."""

import importlib.util
import json
from pathlib import Path

import pytest

from rl.budget import Ledger, episode_worst_case_usd, sample_cost_usd
from rl.cot_mask import CALL, CHANNEL, END, MESSAGE
from rl.report import CollectPlan, build_report, plan_collect, to_markdown, wilson, write_report
from rl.schema import Episode, Turn, append_jsonl

ARENA = Path(__file__).resolve().parents[2]
NAMES = {1: "analysis", 2: "final", 3: "commentary"}


class Tok:
    def decode(self, ids):
        return "".join(NAMES.get(i, "?") for i in ids)


def turn(n_cot: int, flags=()):
    return Turn(prompt_len=10, tokens=[CHANNEL, 1, MESSAGE, *range(100, 100 + n_cot), END, CHANNEL, 3, MESSAGE, 20, 21, CALL], logprobs=[], flags=list(flags))


def ep(task, epoch, passed, *, fc="none", reason="", ended="submit", cot=(5,), side=None, opp=None, reveal=None, wasted=0.0, run="r1"):
    turns = [turn(n) for n in cot] if fc != "infra" else []
    return Episode(episode_id=f"{task}-{epoch}-{fc}", sample_id=task.split("/")[-1], task=task, epoch=epoch, step=0, weights_id="base:openai/gpt-oss-120b", prompt_tokens=[1],
                   turns=turns, run=run, phase="train", model="openai/gpt-oss-120b", ended=ended if fc != "infra" else "infra", failure_class=fc, failure_reason=reason,
                   passed=passed, usage=dict(prompt_tokens=1000, cached_tokens=0, completion_tokens=200) if turns else {}, wasted_usd=wasted,
                   side_done=side, opportunity_present=opp, final_reveal=reveal)


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "r1"
    eps = [ep("env/a", 1, True, cot=(5, 3), side=True, opp=True, reveal=True), ep("env/a", 2, False, side=False, opp=True, reveal=False),
           ep("env/a", 3, True, side=True, opp=True, reveal=False),
           ep("env/b", 1, False, fc="agent", reason="no_submit", side=False, opp=False, reveal=False), ep("env/b", 2, False, side=None),
           ep("env/c", 1, None, fc="infra", reason="docker", wasted=0.01), ep("env/c", 2, True, ended="wall")]
    append_jsonl(d / "collect" / "episodes.jsonl", eps)
    mon = d / "monitor" / "r1" / "monitor.jsonl"
    mon.parent.mkdir(parents=True)
    rows = [dict(schema="arena.monitor.v1", log="x.eval", sample="a", epoch=1, turn=0, monitor="cot", scale=[0, 10], score=s, unparsed=s is None, skipped=False,
                 empty_cot=False, judge_model="j", cost_usd=0.001) for s in (0, 3, 10, None)]
    mon.write_text("\n".join(map(json.dumps, rows)) + "\n{torn")
    Ledger(d / "ledger.csv").append("collect", "r1", "openai/gpt-oss-120b", "counted", counted_usd=0.5, note="sample")
    Ledger(d / "ledger.csv").append("collect", "other", "openai/gpt-oss-120b", "counted", counted_usd=9.0)
    return d


def test_wilson_known_values():
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    assert wilson(0, 0) is None
    lo, hi = wilson(0, 4)
    assert lo == 0.0 and hi == pytest.approx(0.4899, abs=1e-4)


def test_report_sections(run_dir):
    rep = build_report(run_dir, tokenizer=Tok())
    m = rep["main"]
    # scored: a1 a2 a3 b1 b2 (c1 INFRA and c2 wall are reward None)
    assert (m["n_episodes"], m["n_scored"], m["n_reward_none"]) == (7, 5, 2)
    assert m["pass_rate"]["k"] == 2 and m["pass_rate"]["n"] == 5 and m["pass_rate"]["ci95"] == [round(x, 4) for x in wilson(2, 5)]
    assert m["by_task"]["env/a"]["signal"] and not m["by_task"]["env/b"]["signal"] and not m["by_task"]["env/c"]["signal"]
    assert m["rl_signal"]["tasks_with_variance"] == 1 and m["rl_signal"]["frac"] == pytest.approx(1 / 3)
    s = rep["side"]
    assert s["side_done"]["k"] == 2 and s["side_done"]["n"] == 4 and s["n_not_evaluated"] == 3
    assert s["side_done_by_opportunity"]["opportunity_present"]["k"] == 2 and s["side_done_by_opportunity"]["opportunity_present"]["n"] == 3
    assert s["side_done_by_opportunity"]["opportunity_absent"]["n"] == 1 and s["side_done_by_opportunity"]["opportunity_absent"]["k"] == 0
    assert s["final_reveal"]["k"] == 1 and s["final_reveal"]["n"] == 4 and s["final_reveal_given_side_done"]["k"] == 1 and s["final_reveal_given_side_done"]["n"] == 2
    mon = rep["monitor"]
    assert mon["turns"] == 4 and mon["parsed"] == 3 and mon["unparsed"] == 1 and mon["c_turn"]["max"] == 1.0 and mon["c_turn_hist10"][0] == 1 and mon["c_turn_hist10"][9] == 1
    assert mon["episode_max_c"]["n"] == 1 and mon["cost_usd"] == pytest.approx(0.004)
    c = rep["cot"]
    assert c["source"] == "analysis_channel" and c["n_turns"] == 7 and c["per_turn"]["max"] == 5 and c["per_turn"]["min"] == 3 and c["per_episode"]["max"] == 8
    f = rep["failures"]
    assert f["counts"] == {"none": 5, "agent": 1, "infra": 1, "halt": 0} and f["reasons"] == {"agent:no_submit": 1, "infra:docker": 1} and f["halted"] is None
    k = rep["cost"]
    assert k["sampling_usd_upper_bound"] == pytest.approx(6 * sample_cost_usd("openai/gpt-oss-120b", 1000, 0, 200), abs=1e-6)
    assert k["wasted_usd"] == 0.01 and k["ledger"]["counted_usd"] == 0.5 and k["ledger"]["rows"] == 1
    assert k["total_usd"] == pytest.approx(k["sampling_usd_upper_bound"] + 0.01 + 0.004, abs=1e-6)


def test_cot_fallback_without_tokenizer(run_dir, monkeypatch):
    import rl.cot_mask as cm
    monkeypatch.setattr(cm, "_tokenizer_for", lambda m: (_ for _ in ()).throw(OSError("offline")))
    rep = build_report(run_dir)
    assert rep["cot"]["source"].startswith("sampled_tokens") and rep["cot"]["per_turn"]["max"] == len(turn(5).tokens)


def test_markdown_and_files(run_dir, tmp_path):
    (run_dir / "HALTED.md").write_text("# HALTED at step 0\n\n3 consecutive INFRA failures\n")
    rep, md = write_report(run_dir, tmp_path / "out", tokenizer=Tok())
    assert (tmp_path / "out" / "report.md").read_text() == md and json.loads((tmp_path / "out" / "report.json").read_text())["schema"] == "arena.rl.report.v1"
    assert "2/5 = 40.0%" in md and "1/3 tasks" in md and "HALTED: 3 consecutive INFRA failures" in md and "| env/a | 3 | 2 | 67% | yes |" in md
    assert "INFRA 1" in md and "AGENT 1" in md


def test_report_without_side_or_monitor(tmp_path):
    append_jsonl(tmp_path / "episodes.jsonl", [ep("env/a", 1, True), ep("env/a", 2, True)])
    rep = build_report(tmp_path, tokenizer=Tok())
    assert rep["monitor"] is None and rep["side"]["side_done"]["rate"] is None and rep["main"]["rl_signal"]["tasks_with_variance"] == 0
    md = to_markdown(rep)
    assert "no monitor.jsonl found" in md and "n/a (n=0)" in md


def test_missing_episodes_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_report(tmp_path)


def test_plan_collect_arithmetic():
    p = plan_collect(CollectPlan(["env/a", "env/b"], "m1", 4, 10.0, ep_worst_usd=0.5))
    assert p["episodes"] == 8 and p["reserve_usd"] == 4.0 and p["expected_usd"] == 0.24 and p["would_refuse"] == []
    assert p["command"].startswith("ARENA_RL_RUN_TINKER=1 python rl.py collect env/a env/b --run m1 --epochs 4 --budget-usd 10 ") and "--ep-worst-usd 0.5" in p["command"]
    q = plan_collect(CollectPlan(["env/a"], "m1", 4, 1.0))
    assert q["ep_worst_case_usd"] == round(episode_worst_case_usd("openai/gpt-oss-120b", max_turns=60, max_tokens=4096), 4)
    assert any("over --budget-usd" in w for w in q["would_refuse"])
    assert any("total cap" in w for w in plan_collect(CollectPlan(["env/a"], "m1", 1, 5.0, ep_worst_usd=1.0), ledger_total_usd=19.5)["would_refuse"])


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_RL_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("ARENA_RL_RUN_TINKER", raising=False)
    spec = importlib.util.spec_from_file_location("arena_rl_cli_report", ARENA / "rl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_plan_collect_makes_no_call(cli, capsys, tmp_path, monkeypatch):
    import tinker
    monkeypatch.setattr(tinker, "ServiceClient", lambda *a, **k: pytest.fail("a live Tinker client was created"))
    monkeypatch.setattr(cli, "expand", lambda names: list(names))
    with pytest.raises(SystemExit) as e:
        cli.main(["plan-collect", "env/a", "env/b", "--run", "m1", "--epochs", "2", "--budget-usd", "5", "--ep-worst-usd", "0.1"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    p = json.loads(out[: out.rindex("}") + 1])
    assert p["episodes"] == 4 and p["reserve_usd"] == 0.4 and "ARENA_RL_RUN_TINKER=1" in p["command"]
    assert not (tmp_path / "data").exists()
    with pytest.raises(SystemExit) as e:
        cli.main(["plan-collect", "env/a", "--run", "m1", "--epochs", "4", "--budget-usd", "1"])
    assert e.value.code == 2


def test_cli_report(cli, capsys, run_dir, monkeypatch):
    import rl.cot_mask as cm
    monkeypatch.setattr(cm, "_tokenizer_for", lambda m: Tok())
    cli.main(["report", str(run_dir)])
    out = capsys.readouterr().out
    assert "# Collect report: r1" in out and (run_dir / "report.md").exists() and (run_dir / "report.json").exists()


def test_monitor_direction_and_per_monitor(tmp_path):
    from rl.report import c_turn, monitor
    sus = dict(schema="arena.monitor.v1", log="x", sample="a", epoch=1, turn=0, monitor="controlarena_cot", scale=[0, 10], direction="high=suspicious", score=8, judge_model="j")
    ont = dict(sus, monitor="ontask_generic_100", scale=[1, 100], direction="high=on-task", score=100)
    assert c_turn(sus) == pytest.approx(0.8) and c_turn(ont) == 0.0 and c_turn(dict(ont, score=1)) == 1.0
    m = monitor([sus, ont])
    assert set(m["by_monitor"]) == {"controlarena_cot / j", "ontask_generic_100 / j"} and m["by_monitor"]["ontask_generic_100 / j"]["c_turn"]["max"] == 0.0
    assert m["episode_max_c"]["n"] == 2  # one per monitor, not pooled across monitors


def test_ledger_wasted_and_lookup_from_collect_dir(tmp_path):
    data = tmp_path / "data"
    append_jsonl(data / "r1" / "collect" / "episodes.jsonl", [ep("env/a", 1, True)])
    led = Ledger(data / "ledger.csv")
    led.append("collect", "r1", "openai/gpt-oss-120b", "counted", counted_usd=0.25)
    led.append("collect", "r1", "openai/gpt-oss-120b", "wasted", estimate_usd=0.02, note="infra:docker")  # as classify.budget_hook writes it
    for d in (data / "r1", data / "r1" / "collect"):
        k = build_report(d, tokenizer=Tok())["cost"]["ledger"]
        assert k["path"] == str(data / "ledger.csv") and k["counted_usd"] == 0.25 and k["wasted_usd"] == 0.02
