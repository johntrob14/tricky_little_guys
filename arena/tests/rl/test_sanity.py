"""rl.sanity: the shared battery on base + checkpoints, with fake sampler and judge; the live path is gated."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from rl.budget import Budget, BudgetExceeded, Ledger
from rl.sanity import SanityCfg, Target, battery, fake_gen, fake_judge, run_sanity, table, targets, worst_case_usd

ARENA = Path(__file__).resolve().parents[2]


def ckpts(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rows = [dict(name="r-s001-x-state", kind="state", path="tinker://a/weights/s1", step=1),
            dict(name="r-s003-x-state", kind="state", path="tinker://a/weights/s3", step=3),
            dict(name="r-s003-x", kind="sampler", path="tinker://a/sampler_weights/s3", step=3)]
    (out / "checkpoints.json").write_text(json.dumps(rows))


def budget(tmp_path, cap=1.0):
    return Budget(cap, stage="sanity", run="r", ledger=Ledger(tmp_path / "ledger.csv"))


def test_battery_is_the_shared_module():
    assert Path(battery.__file__).resolve() == ARENA.parent / "sanity" / "battery.py"


def test_targets_base_first_and_dedupe_state_with_sampler(tmp_path):
    ckpts(tmp_path)
    ts = targets(SanityCfg(out_dir=tmp_path))
    assert [(t.name, t.kind) for t in ts] == [("base", "base"), ("r-s001-x-state", "state"), ("r-s003-x", "sampler")]
    assert [t.name for t in targets(SanityCfg(out_dir=tmp_path, kinds=("sampler",)))] == ["base", "r-s003-x"]
    assert [t.name for t in targets(SanityCfg(out_dir=tmp_path / "none"))] == ["base"]


def test_targets_refuse_checkpoints_of_another_model(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "checkpoints.json").write_text(json.dumps([dict(name="c", kind="sampler", path="tinker://x", step=0, model="Qwen/Qwen3-8B"),
                                                           dict(name="d", kind="sampler", path="tinker://y", model="openai/gpt-oss-120b")]))
    with pytest.raises(ValueError, match="Qwen"):
        targets(SanityCfg(out_dir=tmp_path))
    (tmp_path / "checkpoints.json").write_text(json.dumps([dict(name="c", kind="sampler", path="tinker://x", step=0), dict(name="d", kind="sampler", path="tinker://y")]))
    assert [t.name for t in targets(SanityCfg(out_dir=tmp_path))] == ["base", "d", "c"]  # a row without a step sorts first, no TypeError


def test_cli_rejects_bad_kinds(cli):
    with pytest.raises(SystemExit, match="--kinds"):
        cli.main(["sanity", "--run", "r", "--budget-usd", "1", "--fake", "--kinds", "samplr"])


def test_run_writes_beside_each_checkpoint_and_table(tmp_path):
    ckpts(tmp_path)
    cfg = SanityCfg(out_dir=tmp_path)
    res = asyncio.run(run_sanity(cfg, fake_gen({"r-s003-x"}), fake_judge(), budget=budget(tmp_path)))
    root = tmp_path / "sanity"
    nq = len(battery.BATTERY) * cfg.n
    assert len((root / "base" / "answers.jsonl").read_text().splitlines()) == nq
    for name in ("r-s001-x-state", "r-s003-x"):
        assert len((root / name / "judged.jsonl").read_text().splitlines()) == nq
    by = {r["target"]: r for r in res["table"]}
    assert by["base"]["degraded_frac"] is None
    assert by["r-s001-x-state"]["degraded_frac"] == 0.0
    assert by["r-s003-x"]["degraded_frac"] == 1.0 and by["r-s003-x"]["broken_frac"] == 1.0
    assert by["r-s003-x"]["truncated_frac_delta"] == 1.0 and by["r-s003-x"]["repetitive_frac"] == 1.0
    assert "| r-s003-x | 3 | sampler |" in (root / "table.md").read_text()
    # the A/B order was randomised and mapped back
    j = [json.loads(l) for l in (root / "r-s003-x" / "judged.jsonl").read_text().splitlines()]
    assert {r["organism_is_a"] for r in j} == {True, False} and {r["verdict"] for r in j} == {"organism_broken"}


def test_rerun_only_adds_new_checkpoints(tmp_path):
    ckpts(tmp_path)
    cfg = SanityCfg(out_dir=tmp_path, kinds=("state",))
    asyncio.run(run_sanity(cfg, fake_gen(), fake_judge(), budget=budget(tmp_path)))
    calls = []

    def gen_counting():
        g = fake_gen()

        async def gen(t, q, n):
            calls.append(t.name)
            return await g(t, q, n)
        return gen

    res = asyncio.run(run_sanity(SanityCfg(out_dir=tmp_path), gen_counting(), fake_judge(), budget=budget(tmp_path)))
    assert res["ran"] == ["r-s003-x"] and set(calls) == {"r-s003-x"}
    assert {r["target"] for r in res["table"]} == {"base", "r-s001-x-state", "r-s003-x"}


def test_budget_refuses_before_any_sampling(tmp_path):
    ckpts(tmp_path)
    cfg = SanityCfg(out_dir=tmp_path)
    assert worst_case_usd(cfg, 3, 2) > 1e-4
    called = []

    async def gen(t, q, n):
        called.append(1)
        return []
    with pytest.raises(BudgetExceeded):
        asyncio.run(run_sanity(cfg, gen, fake_judge(), budget=budget(tmp_path, cap=1e-4)))
    assert not called


def test_counted_cost_is_ledgered(tmp_path):
    asyncio.run(run_sanity(SanityCfg(out_dir=tmp_path), fake_gen(), fake_judge(), budget=(b := budget(tmp_path))))
    assert b.committed > 0 and b.reserved == 0
    assert any(r["kind"] == "counted" and "sanity_sample" in r["note"] for r in Ledger(tmp_path / "ledger.csv").rows())


def test_table_deltas_vs_base():
    s = [dict(target="base", signals=battery.signals([dict(answer="ok", finish="stop")])),
         dict(target="c", step=1, kind="sampler", n_judged=3, degraded_frac=0.5, broken_frac=0.0, signals=battery.signals([dict(answer="ok", finish="stop")]))]
    rows, md = table(s)
    assert rows[1]["truncated_frac_delta"] == 0.0 and "| c | 1 | sampler | 3 | 0.50 |" in md


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_RL_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("ARENA_RL_RUN_TINKER", raising=False)
    spec = importlib.util.spec_from_file_location("arena_rl_cli_sanity", ARENA / "rl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_live_refused_without_allow_live_and_gate(cli, monkeypatch):
    import tinker
    monkeypatch.setattr(tinker, "ServiceClient", lambda *a, **k: pytest.fail("a live Tinker client was created"))
    with pytest.raises(SystemExit, match="--allow-live"):
        cli.main(["sanity", "--run", "r", "--budget-usd", "1"])
    with pytest.raises(SystemExit, match="ARENA_RL_RUN_TINKER"):
        cli.main(["sanity", "--run", "r", "--budget-usd", "1", "--allow-live"])


def test_cli_budget_mandatory(cli):
    with pytest.raises(SystemExit) as e:
        cli.main(["sanity", "--run", "r", "--fake"])
    assert e.value.code == 2


def test_cli_fake_end_to_end(cli, tmp_path, capsys):
    ckpts(tmp_path / "data" / "fake-r")
    cli.main(["sanity", "--run", "fake-r", "--budget-usd", "1", "--fake", "--fake-broken", "r-s001-x-state"])
    out = capsys.readouterr().out
    assert "| r-s001-x-state | 1 | state |" in out and (tmp_path / "data" / "fake-r" / "sanity" / "table.json").exists()
    assert (tmp_path / "data" / "fake-r" / "sanity" / "ledger.csv").exists() and not (tmp_path / "data" / "ledger.csv").exists()
    rows = {r["target"]: r for r in json.loads((tmp_path / "data" / "fake-r" / "sanity" / "table.json").read_text())}
    assert rows["r-s001-x-state"]["degraded_frac"] == 1.0 and rows["r-s003-x"]["degraded_frac"] == 0.0


def test_cli_dry_run_lists_targets(cli, tmp_path, capsys):
    ckpts(tmp_path / "data" / "r")
    with pytest.raises(SystemExit) as e:
        cli.main(["sanity", "--run", "r", "--budget-usd", "1", "--dry-run"])
    assert e.value.code == 0
    assert json.loads(capsys.readouterr().out)["targets"] == ["base", "r-s001-x-state", "r-s003-x"]
    assert not (tmp_path / "data" / "r" / "sanity").exists()
