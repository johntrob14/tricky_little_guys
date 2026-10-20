"""RL tooling for the arena: preflight, rollouts through the token-recording provider, the training loop, load test. Run as a script (`python rl.py`; the
sibling package `rl/` is what `import rl` finds).

    python rl.py doctor [--no-golden] [--no-guard] [--cleanup] ...          docker/image/disk/memory gates and golden controls, no Tinker
    python rl.py loadtest [--levels 1,4,6,8] ...                            compose load test, no Tinker
    python rl.py collect <tasks> --run R --epochs N --fake [--no-sandbox]   episodes with the scripted fake policy (phase=test); no Tinker
    python rl.py collect <tasks> --run R --epochs N --budget-usd X          real sampling (needs ARENA_RL_RUN_TINKER=1, i.e. John's go)
    python rl.py train <tasks> --run R --steps S --B B --G G --budget-usd X --dry-run     plan and forecast only; never spends
    python rl.py train <tasks> --run fake-R --steps 3 --B 2 --G 4 --budget-usd 1 --fake --no-sandbox     whole loop on fakes
    python rl.py train <tasks> --run R --steps S --B B --G G --budget-usd X               live Tinker (needs ARENA_RL_RUN_TINKER=1)
    python rl.py train ... --beta-cot 0.5 --allow-live [--judge-backend tinker]          + live per-turn CoT-monitor penalty (rl.cot_monitor), judge cost on --budget-usd
    python rl.py sidecheck <config.json> [<fixture_dir>] [--task T] [--final-text S]   validate side_specs; dry-check against a local tree; offline
    python rl.py plan-collect <tasks> --run R --epochs N --budget-usd X     print the gated collect command and its cost estimate; no call
    python rl.py report rl_data/<run> [--monitor F] [--out D]               offline report of a collect dir (report.md + report.json)
    python rl.py sanity --run R --budget-usd X --fake                       sanity battery on base + every checkpoint, offline fakes
    python rl.py sanity --run R --budget-usd X --allow-live                 the same on Tinker (needs ARENA_RL_RUN_TINKER=1 too)

<tasks> are <env> (every task of the env) or <env>/<task>. Live Tinker calls are refused unless ARENA_RL_RUN_TINKER=1; --fake never touches Tinker.
Rollouts reuse run.py's agent solver, task_scorer and build_task unchanged (--guard swaps in rl.guard's hardened scorer).
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

ARENA = Path(__file__).resolve().parent
sys.path.insert(0, str(ARENA))

from rl.budget import Budget, Ledger, episode_worst_case_usd, train_cost_usd  # noqa: E402
from rl.datums import CTX_LIMIT  # noqa: E402

MODEL = "openai/gpt-oss-120b"
GATE = "ARENA_RL_RUN_TINKER"
OUT_ROOT = Path(os.environ.get("ARENA_RL_DATA", ARENA / "rl_data"))


def gate(what: str) -> None:
    if os.environ.get(GATE) != "1":
        raise SystemExit(f"refused: {what} would call Tinker (and spend). Set {GATE}=1 only with John's go; use --fake or --dry-run otherwise.")


def expand(names: list[str]) -> list[str]:
    from rl.run_rl_eval import select_tasks
    return [f"{p.parent.name}/{p.name}" for p in select_tasks(names)]


def builder(args):
    if getattr(args, "no_sandbox", False):
        from rl.fake_policy import fake_build_task
        return fake_build_task
    if getattr(args, "guard", False):
        from rl.guard import guarded_build_task
        return guarded_build_task
    from run import build_task
    return build_task


def fake_policy(args):
    from rl.collect import ensure_hf_cache
    from rl.fake_policy import ScriptedPolicy
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    ensure_hf_cache(args.model)
    return ScriptedPolicy(get_tokenizer(args.model), default_p=args.p_fix, seed=args.seed)


def worst_episode_usd(args) -> float:
    if getattr(args, "ep_worst_usd", None):
        return args.ep_worst_usd
    return episode_worst_case_usd(args.model, max_turns=args.max_turns, max_tokens=args.max_tokens)


def cmd_doctor(args, rest):
    from rl import doctor
    sys.argv = ["rl.py doctor", *rest]
    doctor.main()


def cmd_loadtest(args, rest):
    from rl import loadtest
    sys.argv = ["rl.py loadtest", *rest]
    loadtest.main()


async def collect_async(args):
    from rl.collect import Collector
    from rl.loop import CollectRequest
    from rl.rewards import attach_reward, compute_rewards
    from rl.schema import append_jsonl

    tasks = expand(args.tasks)
    out = OUT_ROOT / args.run
    out.mkdir(parents=True, exist_ok=True)
    if args.fake:
        sampler, wid = fake_policy(args), "fake"
    else:
        gate("collect")
        import tinker
        from rl.train import new_service
        svc = new_service(args.run, "collect")
        sampler = svc.create_sampling_client(model_path=args.model_path) if args.model_path else svc.create_sampling_client(base_model=args.model)
        wid = args.model_path or f"base:{args.model}"
    ledger = Ledger(out / "ledger.csv") if args.fake else Ledger()
    budget = Budget(args.budget_usd, args.model, stage="collect", run=args.run, ledger=ledger)
    reserve = worst_episode_usd(args) * len(tasks) * args.epochs
    budget.start_stage(reserve if not args.fake else 0.0, note=f"collect {len(tasks)} tasks x {args.epochs}")
    budget.reserve(reserve)
    collector = Collector(args.run, args.model, max_turns=args.max_turns, max_tokens=args.max_tokens, max_sandboxes=args.max_sandboxes, build=builder(args),
                          phase="test" if args.fake else args.phase, split="train" if not args.fake else "test", mode=args.mode)
    req = CollectRequest(0, tasks, args.epochs, sampler, wid, out / "collect", None, budget)
    t0 = time.time()
    eps = await collector(req)
    eps = [attach_reward(e) if e.passed is None else e for e in eps]
    from rl.loop import episode_sample_usd
    spent = sum(episode_sample_usd(e, args.model) for e in eps)
    budget.release(reserve)
    budget.commit(spent, kind="sample", note=f"collect {len(eps)} episodes")
    append_jsonl(out / "collect" / "episodes.jsonl", eps)
    rewards = compute_rewards(eps)
    summary = dict(run=args.run, seconds=round(time.time() - t0, 1), n=len(eps), by_task={t: dict(n=sum(e.task == t for e in eps), passed=sum(rewards[e.episode_id] == 1.0 for e in eps if e.task == t),
                   infra=sum(e.failure_class == "infra" for e in eps if e.task == t)) for t in tasks}, ended=dict(Counter(e.ended for e in eps)),
                   turns=[len(e.turns) for e in eps], sampled_usd_upper_bound=round(spent, 6), episodes=str(out / "collect" / "episodes.jsonl"))
    print(json.dumps(summary, indent=1))


def cmd_collect(args, rest):
    asyncio.run(collect_async(args))


def cmd_sidecheck(args, rest):
    from rl.side_config import SideSpecError, dry_check, load_side_specs
    try:
        specs = load_side_specs(args.config)
    except SideSpecError as e:
        raise SystemExit(f"invalid side_specs in {args.config}: {e}")
    if args.task:
        if args.task not in specs:
            raise SystemExit(f"--task {args.task!r} has no side spec; have {sorted(specs)}")
        specs = {args.task: specs[args.task]}
    if args.fixture is None:
        print(json.dumps(dict(config=args.config, valid=True, tasks=sorted(specs)), indent=1))
        return
    res = [dry_check(s, args.fixture, task=t, final_text=args.final_text).as_dict() for t, s in specs.items()]
    print(json.dumps(dict(config=args.config, fixture=str(args.fixture), results=res), indent=1))
    if not all(r["consistent"] for r in res):
        raise SystemExit(4)
def cmd_plan_collect(args, rest):
    from rl.report import CollectPlan, plan_collect
    p = plan_collect(CollectPlan(expand(args.tasks), args.run, args.epochs, args.budget_usd, args.model, args.max_turns, args.max_tokens, args.ep_worst_usd,
                                 ep_usd_est=args.ep_usd_est), Ledger(OUT_ROOT / "ledger.csv").total_counted())
    print(json.dumps(p, indent=1))
    print("dry run: no Tinker call, no docker, no ledger row" + ("; the command above would be refused for the reasons listed" if p["would_refuse"] else ""))
    raise SystemExit(2 if p["would_refuse"] else 0)


def cmd_report(args, rest):
    from rl.report import write_report
    d = Path(args.dir)
    if not d.exists() and (OUT_ROOT / args.dir).exists():
        d = OUT_ROOT / args.dir
    rep, md = write_report(d, Path(args.out) if args.out else None, monitor_paths=[Path(m) for m in args.monitor], ledger_path=Path(args.ledger) if args.ledger else None)
    print(md)
    print(f"wrote {Path(args.out) if args.out else d}/report.md and report.json")


def make_cfg(args, tasks):
    from rl.loop import HaltRules, LoopCfg
    from rl.train import TrainHP

    return LoopCfg(run=args.run, out_dir=OUT_ROOT / args.run, train_tasks=tasks, steps=args.steps, B=args.B, G=args.G, ep_worst_usd=worst_episode_usd(args), model=args.model,
                   seed=args.seed, hp=TrainHP(learning_rate=args.lr, seed=args.seed), adv_norm=args.adv_norm, check_direction=args.check_direction, state_every=args.state_every,
                   halt=HaltRules(), reward_cfg=reward_cfg(args))


def reward_cfg(args):
    from rl.rewardcfg import RewardCfg
    beta = getattr(args, "beta_cot", 0.0) or 0.0
    return RewardCfg(beta_cot=beta, cot_penalty_enabled=True) if beta else RewardCfg()


def cot_gate(args, judge=None) -> None:
    """The CoT-monitor penalty calls a live judge on every turn: refuse unless --allow-live (or an injected judge) and a finite positive --budget-usd."""
    if not getattr(args, "beta_cot", 0.0):
        return
    if not (args.beta_cot > 0 and math.isfinite(args.beta_cot)):
        raise SystemExit(f"refused: --beta-cot must be finite and > 0, got {args.beta_cot}")
    if not (math.isfinite(args.budget_usd) and args.budget_usd > 0):
        raise SystemExit(f"refused: the CoT monitor needs a finite --budget-usd > 0, got {args.budget_usd}")
    if judge is None and not args.allow_live:
        raise SystemExit("refused: --beta-cot runs a live CoT-monitor judge on every turn; pass --allow-live (judge cost is charged to --budget-usd)")
    if judge is None and args.judge_backend == "tinker":
        gate("the tinker CoT-monitor judge")


def cot_scorer(args, budget, out_dir, judge=None):
    """rl.cot_monitor.CotMonitorScorer for run_loop(cot_score=...), or None when --beta-cot is 0. The live judge is built only after cot_gate."""
    if not getattr(args, "beta_cot", 0.0):
        return None
    from rl.cot_monitor import CotMonitorScorer, live_judge
    from rl.monitor.pricing import DEFAULT_MODEL
    cot_gate(args, judge)
    model = args.judge_model or DEFAULT_MODEL[args.judge_backend]
    if judge is None:
        judge = live_judge(args.judge_backend, model, args.judge_max_tokens)
    return CotMonitorScorer(judge, budget, backend=args.judge_backend, judge_model=model, concurrency=args.cot_concurrency, max_tokens=args.judge_max_tokens,
                            fallback=args.cot_fallback, out_dir=out_dir)


def plan(args, cfg) -> dict:
    from rl.loop import forecast_step
    per_step_rollout = cfg.ep_worst_usd * cfg.B * cfg.G
    per_step_train = train_cost_usd(args.model, cfg.B * cfg.G * CTX_LIMIT)
    forecast = forecast_step(cfg, None, None)
    ledger_total = Ledger().total_counted()
    p = dict(run=cfg.run, model=cfg.model, tasks=cfg.train_tasks, steps=cfg.steps, B=cfg.B, G=cfg.G, episodes=cfg.steps * cfg.B * cfg.G, max_turns=args.max_turns, max_tokens=args.max_tokens,
             lr=cfg.hp.learning_rate, ep_worst_case_usd=round(cfg.ep_worst_usd, 4), step_rollout_worst_case_usd=round(per_step_rollout, 4),
             step_train_worst_case_usd=round(per_step_train, 4), step_forecast_usd=round(forecast, 4), run_forecast_usd=round(forecast * cfg.steps, 4),
             run_worst_case_usd=round((per_step_rollout + per_step_train) * cfg.steps, 4), budget_usd=args.budget_usd, ledger_counted_usd=round(ledger_total, 4),
             total_cap_usd=20.0, cot_monitor=cot_plan(args), forecast_formula="steps * B * G * (ep_usd_est + train_usd_est), est 0.03 and 0.02 USD per episode, x1.5 before the first step is measured")
    problems = []
    if forecast > args.budget_usd:
        problems.append(f"first-step forecast ${forecast:.4f} is over --budget-usd ${args.budget_usd}")
    if per_step_rollout > args.budget_usd:
        problems.append(f"one step's rollout worst-case reservation ${per_step_rollout:.4f} is over --budget-usd ${args.budget_usd}; the loop would halt before sampling")
    if ledger_total + forecast > 20.0:
        problems.append("ledger total plus forecast is over the 20 USD total cap")
    try:
        cot_gate(args)  # so --dry-run reports a CoT-monitor gate refusal instead of planning a run that would be refused
    except SystemExit as e:
        problems.append(str(e))
    p["would_refuse"] = problems
    return p


def cot_plan(args):
    if not getattr(args, "beta_cot", 0.0):
        return None
    from rl.monitor.pricing import DEFAULT_MODEL, price_for
    model = args.judge_model or DEFAULT_MODEL[args.judge_backend]
    pr = price_for(args.judge_backend, model)
    return dict(beta_cot=args.beta_cot, backend=args.judge_backend, judge_model=model, price=pr.meta(), concurrency=args.cot_concurrency, fallback=args.cot_fallback,
                allow_live=args.allow_live, note="one judge call per non-empty, non-truncated turn (up to 2 on an unparsed reply); each reserves its worst case on --budget-usd, not in the forecast above")


async def train_async(args, cot_judge=None):
    from rl.loop import HaltError, run_loop

    tasks = expand(args.tasks)
    cfg = make_cfg(args, tasks)
    if args.fake:
        if not args.run.startswith("fake-"):
            raise SystemExit("--fake runs must be named fake-*: their data must not be mistaken for a real run")
        from rl.collect import Collector
        from rl.fake_policy import FakeService
        policy = fake_policy(args)
        svc = FakeService(policy)
        ledger = Ledger(cfg.out_dir / "ledger.csv")
    else:
        gate("train")
        from rl.train import new_service
        svc, ledger = new_service(args.run, "train"), Ledger()
    budget = Budget(args.budget_usd, args.model, stage="train", run=args.run, ledger=ledger)
    collect = Collector(args.run, args.model, max_turns=args.max_turns, max_tokens=args.max_tokens, max_sandboxes=args.max_sandboxes, build=builder(args))
    try:
        res = await run_loop(cfg, svc, collect, budget=budget, resume=not args.no_resume, base_model=args.model, cot_score=cot_scorer(args, budget, cfg.out_dir, cot_judge))
    except HaltError as e:
        print(f"HALTED: {e} (see {cfg.out_dir / 'HALTED.md'})")
        raise SystemExit(3)
    print(json.dumps(res, indent=1, default=str))


def cmd_train(args, rest, cot_judge=None):
    cfg = make_cfg(args, expand(args.tasks))
    p = plan(args, cfg)
    print(json.dumps(p, indent=1))
    if args.dry_run:
        print("dry run: nothing started, no Tinker call, no docker, no ledger row" + ("; the real run would be refused for the reasons above" if p["would_refuse"] else ""))
        raise SystemExit(2 if p["would_refuse"] else 0)
    if p["would_refuse"] and not args.fake:
        raise SystemExit("refused: " + "; ".join(p["would_refuse"]))
    cot_gate(args, cot_judge)
    asyncio.run(train_async(args, cot_judge))


def cmd_sanity(args, rest):
    from rl.sanity import SanityCfg, fake_gen, fake_judge, live_gen, live_judge, run_sanity, targets, worst_case_usd

    out = OUT_ROOT / args.run
    kinds = tuple(k for k in args.kinds.split(",") if k)
    if not kinds or set(kinds) - {"sampler", "state"}:
        raise SystemExit(f"--kinds must be a comma list of sampler,state; got {args.kinds!r}")
    cfg = SanityCfg(out_dir=out, model=args.model, judge_model=args.judge_model, kinds=kinds, seed=args.seed)
    ts = targets(cfg)
    worst = worst_case_usd(cfg, len(ts), len(ts) - 1)
    print(json.dumps(dict(run=args.run, targets=[t.name for t in ts], worst_case_usd_all=round(worst, 4), budget_usd=args.budget_usd), indent=1))
    if args.dry_run:
        raise SystemExit(0)
    if args.fake:
        gen, judge, ledger = fake_gen(set(args.fake_broken.split(",")) if args.fake_broken else set()), fake_judge(), Ledger(out / "sanity" / "ledger.csv")
    else:
        if not args.allow_live:
            raise SystemExit("refused: the live sanity battery samples from Tinker; pass --allow-live (and set ARENA_RL_RUN_TINKER=1, John's go), or use --fake / --dry-run")
        gate("sanity")
        from rl.train import new_service
        gen, judge, ledger = live_gen(new_service(args.run, "sanity"), args.model), live_judge(args.judge_model), Ledger()
    budget = Budget(args.budget_usd, args.model, stage="sanity", run=args.run, ledger=ledger)
    res = asyncio.run(run_sanity(cfg, gen, judge, budget=budget))
    print(res["table_md"])
    print(json.dumps(dict(ran=res["ran"], budget=res["budget"], table=str(cfg.root / "table.md")), indent=1))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="rl.doctor; extra flags are passed through", add_help=False)
    sub.add_parser("loadtest", help="rl.loadtest; extra flags are passed through", add_help=False)

    def common(p):
        p.add_argument("tasks", nargs="+", help="<env> or <env>/<task>")
        p.add_argument("--run", required=True)
        p.add_argument("--model", default=MODEL)
        p.add_argument("--budget-usd", type=float, required=True, help="hard cap for this stage; the loop halts before passing it")
        p.add_argument("--max-turns", type=int, default=60)
        p.add_argument("--max-tokens", type=int, default=4096, help="per model call")
        p.add_argument("--max-sandboxes", type=int, default=6)
        p.add_argument("--ep-worst-usd", type=float, help="override the per-episode worst case that is reserved (default: budget.episode_worst_case_usd for max-turns and max-tokens); lowers the safety margin")
        p.add_argument("--guard", action="store_true", help="score with rl.guard's hardened scorer")
        p.add_argument("--fake", action="store_true", help="scripted fake policy and (train) fake Tinker service; no Tinker")
        p.add_argument("--no-sandbox", action="store_true", help="with --fake: docker-free fake tools and scorer (rl.fake_policy)")
        p.add_argument("--p-fix", type=float, default=0.5, help="fake policy: probability that an episode takes the fixing path")
        p.add_argument("--seed", type=int, default=0)

    c = sub.add_parser("collect", help="run episodes, write sidecars and episodes.jsonl")
    common(c)
    c.add_argument("--epochs", type=int, default=1)
    c.add_argument("--mode", default="train", choices=["train", "eval"])
    c.add_argument("--phase", default="train")
    c.add_argument("--model-path", help="tinker:// checkpoint to sample instead of the base model")
    t = sub.add_parser("train", help="the training loop")
    common(t)
    t.add_argument("--steps", type=int, required=True)
    t.add_argument("--B", type=int, required=True, help="tasks per step")
    t.add_argument("--G", type=int, required=True, help="episodes per task per step")
    t.add_argument("--lr", type=float, default=4e-5)
    t.add_argument("--adv-norm", default="none", choices=["none", "std"])
    t.add_argument("--state-every", type=int, default=2)
    t.add_argument("--check-direction", action="store_true")
    t.add_argument("--no-resume", action="store_true")
    t.add_argument("--dry-run", action="store_true", help="print the plan and forecast, then exit without spending")
    t.add_argument("--beta-cot", type=float, default=0.0, help="CoT-monitor penalty weight on analysis-channel tokens (organism term; 0 = off, the default)")
    t.add_argument("--allow-live", action="store_true", help="allow the live CoT-monitor judge (needed with --beta-cot)")
    t.add_argument("--judge-backend", default="openrouter", choices=["openrouter", "tinker"], help="CoT-monitor judge backend (tinker also needs ARENA_RL_RUN_TINKER=1)")
    t.add_argument("--judge-model", default=None, help="default: rl.monitor.pricing.DEFAULT_MODEL for the backend (openrouter: openai/gpt-oss-20b)")
    t.add_argument("--judge-max-tokens", type=int, default=6000)
    t.add_argument("--cot-concurrency", type=int, default=8, help="max concurrent judge calls")
    t.add_argument("--cot-fallback", type=float, default=0.0, help="c_turn for a turn whose judge call failed or stayed unparsed (recorded in cot_monitor.jsonl)")
    s = sub.add_parser("sidecheck", help="validate an organism config's side_specs; with a fixture dir, report whether each side check would fire (offline)")
    s.add_argument("config", help="organism config JSON (e.g. configs/organism_leak_secret.json)")
    s.add_argument("fixture", nargs="?", type=Path, help="local directory standing in for the container root: sandbox /app/x is <fixture>/app/x")
    s.add_argument("--task", help="check only this <env>/<task>")
    s.add_argument("--final-text", help="also report whether this final-answer text trips the reveal markers")
    pc = sub.add_parser("plan-collect", help="dry-run planner for a gated collect: exact command and cost estimate, makes no call")
    pc.add_argument("tasks", nargs="+", help="<env> or <env>/<task>")
    pc.add_argument("--run", required=True)
    pc.add_argument("--epochs", type=int, required=True)
    pc.add_argument("--budget-usd", type=float, required=True)
    pc.add_argument("--model", default=MODEL)
    pc.add_argument("--max-turns", type=int, default=60)
    pc.add_argument("--max-tokens", type=int, default=4096)
    pc.add_argument("--ep-worst-usd", type=float)
    pc.add_argument("--ep-usd-est", type=float, default=0.03, help="expected cost per episode (placeholder; use a measured cost.per_sampled_episode_usd from a report)")
    r = sub.add_parser("report", help="offline report of a collect output dir")
    r.add_argument("dir", help="rl_data/<run> (or <run>/collect, or a run name under ARENA_RL_DATA)")
    r.add_argument("--monitor", action="append", default=[], help="extra monitor.jsonl (any under dir are read already)")
    r.add_argument("--ledger", help="ledger.csv (default: dir/ledger.csv, else the shared one beside dir)")
    r.add_argument("--out", help="where report.md and report.json go (default: dir)")
    sn = sub.add_parser("sanity", help="shared sanity battery on the base model and every checkpoint in <run>/checkpoints.json")
    sn.add_argument("--run", required=True)
    sn.add_argument("--model", default=MODEL)
    sn.add_argument("--judge-model", default=MODEL)
    sn.add_argument("--budget-usd", type=float, required=True)
    sn.add_argument("--kinds", default="sampler,state", help="checkpoint kinds to test; a state is skipped when a sampler holds the same step")
    sn.add_argument("--seed", type=int, default=0)
    sn.add_argument("--allow-live", action="store_true", help="sample and judge on Tinker (also needs ARENA_RL_RUN_TINKER=1)")
    sn.add_argument("--fake", action="store_true", help="offline fake sampler and judge")
    sn.add_argument("--fake-broken", default="", help="with --fake: comma-separated checkpoint names that answer degenerately")
    sn.add_argument("--dry-run", action="store_true", help="list targets and the worst case, then exit")
    return ap


def main(argv=None, cot_judge=None):
    ap = parser()
    args, rest = ap.parse_known_args(argv)
    if args.cmd not in ("doctor", "loadtest") and rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    if args.cmd == "train":
        return cmd_train(args, rest, cot_judge)
    {"doctor": cmd_doctor, "loadtest": cmd_loadtest, "collect": cmd_collect, "sidecheck": cmd_sidecheck, "plan-collect": cmd_plan_collect, "report": cmd_report, "sanity": cmd_sanity}[args.cmd](args, rest)


if __name__ == "__main__":
    main()
