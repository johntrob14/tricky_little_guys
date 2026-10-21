"""rl.py --organism-config: side_specs reach Collector and reward_cfg reaches LoopCfg, flag precedence, refusals, and the no-flag path unchanged.

Collector, run_loop and the fake policy are replaced by recorders, so nothing here needs docker, a tokenizer, Tinker or a judge. The goldens in
golden/ are the dry-run stdout of rl.py before --organism-config existed (same commands, empty ARENA_RL_DATA)."""

import importlib.util
import json
from pathlib import Path

import pytest

from rl.rewardcfg import RewardCfg, reward_cfg_from_dict
from rl.side import SideSpec
from rl.side_config import SideSpecError, load_organism

ARENA = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).resolve().parent / "golden"
TASK = "tiktok/missing_user_id_in_guess_response"
TRAIN = ["train", "tiktok", "--steps", "1", "--B", "1", "--G", "2", "--ep-worst-usd", "0.02", "--budget-usd", "1"]
COLLECT_KW = {"max_turns", "max_tokens", "max_sandboxes", "build"}


@pytest.fixture
def cli(tmp_path, monkeypatch):
    import rl.budget
    monkeypatch.setenv("ARENA_RL_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(rl.budget, "DATA_DIR", tmp_path / "data")   # imported earlier with the real rl_data; the goldens assume an empty ledger
    monkeypatch.delenv("ARENA_RL_RUN_TINKER", raising=False)
    spec = importlib.util.spec_from_file_location("arena_rl_cli", ARENA / "rl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def rec(cli, monkeypatch):
    """Record Collector(...) and run_loop(cfg, ...) calls; the fake policy/service are dummies (no tokenizer)."""
    import rl.collect
    import rl.fake_policy
    import rl.loop
    seen = {}

    class Coll:
        def __init__(self, *a, **k):
            seen["collector"] = (a, k)

        async def __call__(self, req):
            return []

    async def run_loop(cfg, svc, collect, **k):
        seen["cfg"], seen["loop_kw"] = cfg, k
        return {}

    monkeypatch.setattr(rl.collect, "Collector", Coll)
    monkeypatch.setattr(rl.loop, "run_loop", run_loop)
    monkeypatch.setattr(rl.fake_policy, "FakeService", lambda p: object())
    monkeypatch.setattr(cli, "fake_policy", lambda args: object())
    return seen


def write(tmp_path, cfg, name="org.json"):
    p = tmp_path / name
    p.write_text(json.dumps(cfg))
    return str(p)


def toy(**rc):
    spec = dict(kind="value_propagated", source="/app/toy_src.txt", target="/app/toy_dst.txt", reveal_markers=["toy_src"], opportunity_present=True)
    return dict(train_tasks=[TASK], reward_cfg=dict(w_main=1.0, w_side=1.0, w_final=0.5, side_enabled=True, **rc), side_specs={TASK: spec})


def dry(cli, capsys, argv):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    out = capsys.readouterr().out
    return e.value.code, out, json.loads(out.split("\ndry run")[0])


def test_no_flag_dry_run_is_byte_identical_to_before(cli, capsys):
    for argv, g in ((["train", "tiktok", "--run", "fake-test-dry", "--steps", "3", "--B", "2", "--G", "4", "--budget-usd", "1", "--ep-worst-usd", "0.02", "--dry-run"],
                     "train_dry_run_default.txt"),
                    (["train", "tiktok", "--run", "fake-test-dry", "--steps", "1", "--B", "1", "--G", "2", "--budget-usd", "1", "--ep-worst-usd", "0.02", "--dry-run",
                      "--beta-cot", "0.5"], "train_dry_run_beta_cot.txt")):
        with pytest.raises(SystemExit):
            cli.main(argv)
        assert capsys.readouterr().out == (GOLDEN / g).read_text()


def test_no_flag_collector_and_loop_unchanged(cli, rec):
    cli.main([*TRAIN, "--run", "fake-test-w", "--fake", "--no-sandbox"])
    assert set(rec["collector"][1]) == COLLECT_KW and rec["cfg"].reward_cfg == RewardCfg() and rec["loop_kw"]["cot_score"] is None
    cli.main([*TRAIN, "--run", "fake-test-w", "--fake", "--no-sandbox", "--guard"])
    assert set(rec["collector"][1]) == COLLECT_KW  # --guard alone still only picks the builder


def test_no_flag_collect_unchanged(cli, rec, capsys):
    cli.main(["collect", "tiktok", "--run", "c", "--fake", "--budget-usd", "1", "--ep-worst-usd", "0.02"])
    assert set(rec["collector"][1]) == COLLECT_KW | {"phase", "split", "mode"}
    assert "side" not in json.loads(capsys.readouterr().out)


def test_config_reaches_collector_and_loop(cli, rec, tmp_path):
    c = write(tmp_path, toy())
    cli.main([*TRAIN, "--run", "fake-test-w", "--fake", "--guard", "--organism-config", c])
    k = rec["collector"][1]
    assert k["guard"] is True and list(k["side_specs"]) == [TASK] and isinstance(k["side_specs"][TASK], SideSpec)
    assert k["side_specs"][TASK].source == "/app/toy_src.txt" and k["side_specs"][TASK].reveal_markers == ["toy_src"]
    assert rec["cfg"].reward_cfg == RewardCfg(w_main=1.0, w_side=1.0, w_final=0.5, side_enabled=True) and rec["loop_kw"]["cot_score"] is None


def test_collect_gets_side_specs(cli, rec, tmp_path, capsys):
    c = write(tmp_path, toy())
    cli.main(["collect", "tiktok", "--run", "c", "--fake", "--budget-usd", "1", "--ep-worst-usd", "0.02", "--organism-config", c])
    assert list(rec["collector"][1]["side_specs"]) == [TASK] and rec["collector"][1]["guard"] is False
    assert json.loads(capsys.readouterr().out)["side"] == {TASK: dict(side_done=0, final_reveal=0)}


def test_config_cot_penalty_is_gated_like_beta_cot(cli, rec, tmp_path):
    import rl.cot_monitor as cm
    cli_cfg = write(tmp_path, toy(beta_cot=0.3, cot_penalty_enabled=True))
    real = cm.live_judge
    cm.live_judge = lambda *a, **k: pytest.fail("a live judge was built")
    try:
        with pytest.raises(SystemExit, match="--allow-live"):
            cli.main([*TRAIN, "--run", "fake-test-w", "--fake", "--organism-config", cli_cfg])
    finally:
        cm.live_judge = real

    async def judge(prompt):
        return "<score>1</score>", 1, 1

    cli.main([*TRAIN, "--run", "fake-test-w", "--fake", "--organism-config", cli_cfg], cot_judge=judge)
    assert rec["cfg"].reward_cfg.beta_cot == 0.3 and rec["cfg"].reward_cfg.cot_penalty_enabled and rec["loop_kw"]["cot_score"] is not None


def test_flag_precedence_in_plan(cli, capsys, tmp_path):
    c = write(tmp_path, toy(beta_cot=0.3, cot_penalty_enabled=True))
    _, _, p = dry(cli, capsys, [*TRAIN, "--run", "r", "--dry-run", "--organism-config", c])
    o = p["organism"]
    assert o["reward_cfg"]["beta_cot"] == 0.3 and o["reward_sources"]["beta_cot"] == "config" and p["cot_monitor"]["beta_cot"] == 0.3
    assert o["side_tasks"] == {TASK: dict(kind="value_propagated", opportunity_present=True)} and o["config_train_tasks_match"] is False and not o["main_only"]
    _, _, p = dry(cli, capsys, [*TRAIN, "--run", "r", "--dry-run", "--organism-config", c, "--beta-cot", "0", "--w-side", "2", "--w-main", "0.5"])
    o = p["organism"]
    assert o["reward_cfg"] == dict(w_main=0.5, w_side=2.0, w_final=0.5, beta_cot=0.0, side_enabled=True, cot_penalty_enabled=False)
    assert o["reward_sources"] == dict(w_main="flag", w_side="flag", w_final="config", beta_cot="flag") and p["cot_monitor"] is None
    _, _, p = dry(cli, capsys, ["train", TASK, *TRAIN[2:], "--run", "r", "--dry-run", "--organism-config", c, "--beta-cot", "0.7"])
    assert p["organism"]["reward_cfg"]["beta_cot"] == 0.7 and p["organism"]["config_train_tasks_match"] is True and p["cot_monitor"]["beta_cot"] == 0.7


def test_main_only_config_plans_main_only(cli, capsys, tmp_path):
    c = write(tmp_path, {})
    code, _, p = dry(cli, capsys, [*TRAIN, "--run", "r", "--dry-run", "--organism-config", c])
    assert code == 0 and p["organism"]["main_only"] and p["organism"]["side_tasks"] == {} and p["organism"]["config_train_tasks_match"] is None


def test_template_with_placeholders_refused(cli, tmp_path):
    t = str(ARENA / "configs" / "organism_sidespec_template.json")
    for argv in ([*TRAIN, "--run", "r", "--dry-run"], [*TRAIN, "--run", "fake-test-w", "--fake"], ["collect", "tiktok", "--run", "c", "--fake", "--budget-usd", "1"]):
        with pytest.raises(SystemExit, match="FILL_ME placeholders.*train_tasks\\[0\\]"):
            cli.main([*argv, "--organism-config", t])
    assert not (tmp_path / "data").exists()  # refused before any ledger row or run dir
    c = write(tmp_path, dict(reward_cfg=dict(w_side="FILL_ME:weight")))
    with pytest.raises(SystemExit, match="reward_cfg.w_side"):
        cli.main([*TRAIN, "--run", "r", "--dry-run", "--organism-config", c])


@pytest.mark.parametrize("cfg,match", [
    (dict(reward_cfg=dict(w_side=float("nan"))), "finite"),
    (dict(reward_cfg=dict(w_final=-1)), ">= 0"),
    (dict(reward_cfg=dict(w_side=True)), "finite"),
    (dict(reward_cfg=dict(side_enabled="yes")), "true/false"),
    (dict(reward_cfg=dict(w_typo=1)), "unknown field"),
    (dict(reward_cfg=[1]), "must be an object"),
    (dict(reward_cfg=dict(side_enabled=True, w_side=1)), "side_specs is empty"),
    ([1, 2], "JSON object"),
])
def test_bad_config_refused(cli, tmp_path, cfg, match):
    c = write(tmp_path, cfg)
    with pytest.raises(SystemExit, match=match):
        cli.main([*TRAIN, "--run", "r", "--dry-run", "--organism-config", c])


def test_refusals_on_misuse(cli, tmp_path, capsys):
    with pytest.raises(SystemExit, match="need --organism-config"):
        cli.main([*TRAIN, "--run", "r", "--dry-run", "--w-side", "1"])
    off = write(tmp_path, dict(reward_cfg=dict(beta_cot=0.0)), "off.json")
    with pytest.raises(SystemExit, match="side_enabled"):
        cli.main([*TRAIN, "--run", "r", "--dry-run", "--organism-config", off, "--w-final", "1"])
    with pytest.raises(SystemExit, match="--w-main must be a finite"):
        cli.main([*TRAIN, "--run", "r", "--dry-run", "--organism-config", off, "--w-main", "inf"])
    with pytest.raises(SystemExit, match="invalid --organism-config"):
        cli.main([*TRAIN, "--run", "r", "--dry-run", "--organism-config", str(tmp_path / "missing.json")])
    c = write(tmp_path, toy())
    code, _, p = dry(cli, capsys, ["train", "tiktok/video_url_format_validation", *TRAIN[2:], "--run", "r", "--dry-run", "--organism-config", c])
    assert code == 2 and any("not among this run's tasks" in x for x in p["would_refuse"])
    code, _, p = dry(cli, capsys, [*TRAIN, "--run", "r", "--dry-run", "--organism-config", c, "--no-sandbox"])
    assert code == 2 and any("--no-sandbox" in x for x in p["would_refuse"])
    with pytest.raises(SystemExit, match="--no-sandbox has none"):
        cli.main([*TRAIN, "--run", "fake-test-w", "--fake", "--no-sandbox", "--organism-config", c])
    assert not (tmp_path / "data" / "fake-test-w" / "ledger.csv").exists()


def test_load_organism_and_reward_cfg_from_dict(tmp_path):
    o = load_organism(write(tmp_path, toy()))
    assert list(o.side_specs) == [TASK] and o.reward_cfg.side_enabled and o.train_tasks == [TASK] and len(o.sha256) == 64
    assert reward_cfg_from_dict(None) == RewardCfg() and reward_cfg_from_dict({"_comment": "x"}) == RewardCfg()
    assert load_organism(ARENA / "configs" / "organism_example.json").reward_cfg.beta_cot == 0.3
    with pytest.raises(SideSpecError, match="valid JSON"):
        p = tmp_path / "bad.json"
        p.write_text("{")
        load_organism(p)


def test_config_train_tasks_never_crash_the_plan(cli, capsys, tmp_path):
    # an unknown env used to raise FileNotFoundError out of the dry-run; a missing task used to be dropped silently
    c = write(tmp_path, dict(train_tasks=["nosuchenv", "tiktok/nosuchtask", TASK]))
    code, _, p = dry(cli, capsys, ["train", TASK, *TRAIN[2:], "--run", "r", "--dry-run", "--organism-config", c])
    assert code == 0 and p["organism"]["config_train_tasks_match"] is False
    assert p["organism"]["config_train_tasks_unknown"] == ["nosuchenv", "tiktok/nosuchtask"]
    _, _, p = dry(cli, capsys, [*TRAIN, "--run", "r", "--dry-run", "--organism-config", str(ARENA / "configs" / "organism_example.json")])
    assert p["organism"]["config_train_tasks_unknown"] == ["tiktok/guess_post_incomplete_validation"]
    for bad in ("tiktok", [1], [""]):
        with pytest.raises(SystemExit, match="train_tasks must be a list"):
            cli.main([*TRAIN, "--run", "r", "--dry-run", "--organism-config", write(tmp_path, dict(train_tasks=bad))])
