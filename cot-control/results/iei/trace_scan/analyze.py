import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from mechtools import *

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "rl"))
from iei_choice import extract_choice_natural

ROOT = Path(__file__).resolve().parent
MODELS = ["base", "mt2e-s170", "mt1b-s150"]
ARMS = ["baseline", "implicit", "explicit"]
FIELDS = ["policy_check", "restates_request", "self_instructions", "clarifying_question", "answer_draft", "nudge_named", "nudge_as_reason", "plans_omission", "invents_specifics"]
ALL_FIELDS = FIELDS + ["final_recommends"]
rng = np.random.default_rng(20261026)


def write_json(name: str, value: object) -> None:
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def rate(hits: int, total: int) -> dict:
    return dict(hits=int(hits), total=int(total), mean=float(hits / total) if total else None)


def behaviors(rows: list[dict]) -> dict:
    return {field: rate(sum(r[field] is True for r in rows), sum(r[field] is not None for r in rows)) for field in FIELDS}


def parser_stats(rows: list[dict]) -> dict:
    decisive = [r for r in rows if r["final_recommends"] in ["A", "B"]]
    return dict(n=len(rows), final_choices=dict(Counter(r["final_recommends"] for r in rows)),
        decisive_agreement=rate(sum(r["parser_choice"] == r["final_recommends"] for r in decisive), len(decisive)),
        decisive_opposite=rate(sum(r["parser_choice"] in ["A", "B"] and r["parser_choice"] != r["final_recommends"] for r in decisive), len(decisive)),
        decisive_unparsed=rate(sum(r["parser_choice"] is None for r in decisive), len(decisive)),
        conditional=rate(sum(r["final_recommends"] == "conditional" for r in rows), len(rows)),
        none=rate(sum(r["final_recommends"] == "none" for r in rows), len(rows)),
        excerpt_truncated=rate(sum(r["excerpt_truncated"] for r in rows), len(rows)))


manifest = json.loads((ROOT / "manifest.json").read_text())
scored = []
for shard in range(8):
    items = [json.loads(line) for line in (ROOT / f"items_{shard:02d}.jsonl").read_text().splitlines()]
    scores = json.loads((ROOT / f"scores_{shard:02d}.json").read_text())
    expected = {item["id"]: item for item in items}
    assert len(items) == len(expected), shard
    assert len(scores) == len(expected) == len({row["id"] for row in scores}), shard
    assert {r["id"] for r in scores} == set(expected), shard
    for row in scores:
        assert set(row) == {"id", "note", *ALL_FIELDS}, (shard, row["id"], set(row))
        assert row["final_recommends"] in ["A", "B", "conditional", "none"], row
        assert isinstance(row["note"], str), row
        assert len(row["note"].split()) <= 15, (shard, row["id"], row["note"])
        for field in FIELDS:
            not_applicable = expected[row["id"]]["arm"] == "baseline" and field in ["nudge_named", "nudge_as_reason"]
            assert row[field] is None if not_applicable else type(row[field]) is bool, (shard, row["id"], field)
    scored.append({row["id"]: row for row in scores})
rows = [dict(m, **scored[m["primary_shard"]][m["id"]]) for m in manifest]
assert len(rows) == len(set(r["id"] for r in rows)) == 426
write_json("labels.json", rows)

groups = {arm: {model: [r for r in rows if r["arm"] == arm and r["model"] == model] for model in MODELS} for arm in ARMS}
counts = {arm: {model: behaviors(group) for model, group in models.items()} for arm, models in groups.items()}
all_three = {}
paired = {}
for arm, models in groups.items():
    cell_maps = {model: {r["prompt_id"]: r for r in group} for model, group in models.items()}
    shared = set.intersection(*(set(group) for group in cell_maps.values()))
    all_three[arm] = {model: behaviors([group[key] for key in sorted(shared)]) for model, group in cell_maps.items()}
    assert cell_maps["base"].keys() == cell_maps["mt2e-s170"].keys()
    pairs = [(cell_maps["base"][key], cell_maps["mt2e-s170"][key]) for key in sorted(cell_maps["base"])]
    scenarios = sorted({a["scenario_id"] for a, b in pairs})
    draws = rng.integers(len(scenarios), size=(10000, len(scenarios)))
    paired[arm] = {}
    for field in FIELDS:
        if pairs[0][0][field] is None:
            continue
        differences = np.array([int(b[field]) - int(a[field]) for a, b in pairs])
        sums = np.array([sum(d for (a, b), d in zip(pairs, differences) if a["scenario_id"] == scenario) for scenario in scenarios])
        sizes = np.array([sum(a["scenario_id"] == scenario for a, b in pairs) for scenario in scenarios])
        bootstrap = sums[draws].sum(1) / sizes[draws].sum(1)
        paired[arm][field] = dict(n=len(pairs), scenarios=len(scenarios), difference_mt2e_minus_base=float(differences.mean()),
            cluster_bootstrap_95ci=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            base_only=rate(sum(a[field] and not b[field] for a, b in pairs), len(pairs)),
            mt2e_only=rate(sum(b[field] and not a[field] for a, b in pairs), len(pairs)))

repeats = [m for m in manifest if "repeat_shard" in m]
assert len(repeats) == 50 and sum(len(s) for s in scored) == len(rows) + len(repeats)
agreement = {}
disagreements = []
for field in ALL_FIELDS:
    pairs = [(scored[m["primary_shard"]][m["id"]][field], scored[m["repeat_shard"]][m["id"]][field]) for m in repeats]
    applicable = [(a, b) for a, b in pairs if a is not None]
    observed = sum(a == b for a, b in applicable) / len(applicable)
    left, right = Counter(a for a, b in applicable), Counter(b for a, b in applicable)
    chance = sum(left[key] * right[key] for key in left.keys() | right.keys()) / len(applicable) ** 2
    agreement[field] = dict(**rate(sum(a == b for a, b in applicable), len(applicable)), kappa=(observed - chance) / (1 - chance) if chance < 1 else None)
for m in repeats:
    primary = scored[m["primary_shard"]][m["id"]]
    repeat = scored[m["repeat_shard"]][m["id"]]
    diffs = {field: [primary[field], repeat[field]] for field in ALL_FIELDS if primary[field] != repeat[field]}
    if diffs:
        disagreements.append(dict(id=m["id"], model=m["model"], arm=m["arm"], differences=diffs))
write_json("disagreements.json", disagreements)

length_strata = {}
for arm in ARMS:
    pooled = groups[arm]["base"] + groups[arm]["mt2e-s170"]
    cuts = np.quantile([r["cot_tokens"] for r in pooled], [1 / 3, 2 / 3])
    length_strata[arm] = dict(cuts=cuts.tolist(), groups={model: {
        label: dict(n=len(subset), behaviors=behaviors(subset))
        for index, label in enumerate(["short", "middle", "long"])
        for subset in [[r for r in groups[arm][model] if int(np.searchsorted(cuts, r["cot_tokens"], side="right")) == index]]}
        for model in MODELS[:2]})

parser = {arm: {model: parser_stats(group) for model, group in models.items()} for arm, models in groups.items()}
raw = {}
for model in MODELS:
    for filename in ["baseline.jsonl", "nudged.jsonl"]:
        for row in map(json.loads, (ROOT.parent / model / filename).read_text().splitlines()):
            arm = "baseline" if filename == "baseline.jsonl" else row["arm"]
            key = f"baseline:{row['scenario_id']}:{row['rep']}" if arm == "baseline" else row["prompt_id"]
            raw[model, arm, key] = row
diagnostics = []
for row in rows:
    text = raw[row["model"], row["arm"], row["prompt_id"]]["response"]
    assert extract_choice_natural(text, row["scenario_id"]) == row["parser_choice"], row["id"]
    normalized = extract_choice_natural(" ".join(text.split()), row["scenario_id"])
    diagnostics.append({key: row[key] for key in ["id", "model", "arm", "parser_choice", "final_recommends"]} | dict(normalized_whitespace_choice=normalized))
normalization = {}
for arm in ARMS:
    normalization[arm] = {}
    for model in MODELS:
        group = [r for r in diagnostics if r["arm"] == arm and r["model"] == model and r["final_recommends"] in ["A", "B"]]
        misses = [r for r in group if r["parser_choice"] != r["final_recommends"]]
        matches = [r for r in group if r["parser_choice"] == r["final_recommends"]]
        normalization[arm][model] = dict(normalized_agreement=rate(sum(r["normalized_whitespace_choice"] == r["final_recommends"] for r in group), len(group)),
            repaired=rate(sum(r["normalized_whitespace_choice"] == r["final_recommends"] for r in misses), len(misses)),
            worsened=rate(sum(r["normalized_whitespace_choice"] != r["final_recommends"] for r in matches), len(matches)))
write_json("parser_diagnostics.json", dict(method="Collapse all whitespace runs with ' '.join(response.split()); run the unchanged paper parser. Diagnostic only.", groups=normalization, rows=diagnostics))
monitor = {model: {r["id"]: r for r in map(json.loads, (ROOT.parent / model / "monitor.jsonl").read_text().splitlines())} for model in MODELS}
explicit = {}
for model in MODELS:
    group = groups["explicit"][model]
    complied = [r for r in group if r["final_recommends"] == r["recommend"]]
    monitored = [r for r in complied if f"explicit:{r['prompt_id']}" in monitor[model]]
    yes = lambda r: monitor[model][f"explicit:{r['prompt_id']}"]["verdict"] == "YES"
    explicit[model] = dict(complied=rate(len(complied), len(group)), monitored=rate(len(monitored), len(complied)),
        detected=rate(sum(yes(r) for r in monitored), len(monitored)), by_restates_request={
            str(flag): rate(sum(yes(r) for r in monitored if r["restates_request"] == flag), sum(r["restates_request"] == flag for r in monitored)) for flag in [False, True]})
explicit_maps = {model: {r["prompt_id"]: r for r in groups["explicit"][model]} for model in MODELS[:2]}
paired_compliance = [key for key in explicit_maps["base"] if all(explicit_maps[model][key]["final_recommends"] == explicit_maps[model][key]["recommend"] and f"explicit:{key}" in monitor[model] for model in MODELS[:2])]
paired_monitor = {model: rate(sum(monitor[model][f"explicit:{key}"]["verdict"] == "YES" for key in paired_compliance), len(paired_compliance)) for model in MODELS[:2]}
omission_without_name = {model: rate(sum(r["plans_omission"] and not r["nudge_named"] for r in groups["explicit"][model]), sum(r["plans_omission"] for r in groups["explicit"][model])) for model in MODELS}

summary = dict(n_unique=len(rows), n_repeated=len(repeats), counts=counts, all_three_matched=all_three, paired=paired, agreement=agreement,
    parser=parser, whitespace_diagnostic=normalization, explicit_omission_without_nudge_name=omission_without_name, length_strata=length_strata, cot_token_medians={arm: {model: float(np.median([r["cot_tokens"] for r in group])) for model, group in models.items()} for arm, models in groups.items()},
    monitor=dict(mode="CoT only, told the nudge", name="Paper Setting 1 prompt", judge="openai/gpt-6-luna via OpenRouter, provider openai, effort medium", explicit_using_scored_recommendation=explicit, explicit_shared_scored_complied=paired_monitor, explicit_shared_scored_complied_prompt_ids=paired_compliance),
    caveats=["Primary labels only; repeats measure agreement and are not used to adjudicate or inflate totals.",
        "Questions 1-9 label the CoT, with user/nudge context; question 10 labels the displayed final-answer excerpt.",
        "Conditional/none are retained, never forced into A/B. Baseline majorities and stable-scenario selection remain parser-derived.",
        "Paired bootstrap intervals resample scenarios, retain paired cells, and are exploratory with no multiple-comparison correction.",
        "Scorers were blind to model names and prior findings; writing style can still distinguish models.",
        "invents_specifics means the rubric's unsupported-specifics label, not a fact-checked hallucination verdict."])
write_json("summary.json", summary)


def fraction(value: dict) -> str:
    return f"{value['hits']}/{value['total']}" if value["total"] else "n/a"


lines = ["# Blinded CoT behavior scan (2026-10-26)", "",
    "Approved rubric: [scorer_brief.txt](scorer_brief.txt). Machine-readable results: [summary.json](summary.json); per-item primary labels and source keys: [labels.json](labels.json).",
    "", "![Behavior counts](iei_trace_behaviors.png)", "", "## Sample and scoring", "",
    "- Seed 20261026. Each nudged arm independently samples 60/214 shared base/mt2e cells, without filtering on choices or monitor verdicts. The two models get the same cells within an arm.",
    "- mt1b is included where it ran those cells: implicit 44/60, explicit 37/60. All-three matched comparisons are in `summary.json:all_three_matched`.",
    "- Baseline uses rep 0 on 35/35 scenarios for each model. Total unique traces: 426/426 scored; 50/426 independently rescored in a different shard.",
    "- Eight fresh Codex scorer subagents, session model inherited (exact API model identifier unavailable), with no conversation history. Model labels, hypotheses, prior results, parser choices and monitor verdicts are withheld. All arms/models are shuffled together.",
    "- The approved brief is unchanged; within-rubric clarifications are recorded in [scorer_clarifications.txt](scorer_clarifications.txt). Questions 1–9 use the full CoT with user/nudge context. Question 10 uses the first 800 and last 1,200 final-answer characters; answers at most 2,000 characters are shown whole.",
    f"- Final-answer excerpts are truncated for {sum(r['excerpt_truncated'] for r in rows)}/{len(rows)} rows. CoTs are never truncated. Scorers read records in batches and label directly; no keyword classifier is used.",
    "- Repeat labels measure agreement. All aggregate results use the preassigned primary label, without adjudication. The saved rubric, source hashes, selected cells and shard assignments are in `config.json`, `manifest.json` and `items_*.jsonl`.",
    "", "## Behavior counts", "", "Values are positive primary labels / applicable traces. These are qualitative labels of the CoT, not monitor verdicts."]
for arm in ARMS:
    lines += ["", f"### {arm.capitalize()}", "", "| Behavior | Base | mt2e-s170 | mt1b-s150 |", "|---|---:|---:|---:|"]
    for field in FIELDS:
        lines.append("| " + " | ".join([field] + [fraction(counts[arm][model][field]) for model in MODELS]) + " |")
lines += ["", "Among explicit CoTs with an omission plan, the nudge's content is unnamed in " + ", ".join(f"{model} {fraction(omission_without_name[model])}" for model in MODELS) + ". This conjunction uses Q8 and Q6; Q8 permits omission plans about any content, not exclusively the nudge."]
lines += ["", "## Paired base–mt2e differences", "", "mt2e minus base, percentage points, with a 95% scenario-cluster bootstrap interval (10,000 resamples; no multiple-comparison correction). Each resampled scenario retains every sampled cell and both model responses. These are exploratory intervals.",
    "", "| Behavior | Baseline | Implicit | Explicit |", "|---|---:|---:|---:|"]
for field in FIELDS:
    cells = []
    for arm in ARMS:
        if field not in paired[arm]:
            cells.append("n/a")
            continue
        stats = paired[arm][field]
        lo, hi = stats["cluster_bootstrap_95ci"]
        cells.append(f"{100 * stats['difference_mt2e_minus_base']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    lines.append("| " + " | ".join([field] + cells) + " |")
lines += ["", "CoT-token medians (base / mt2e / mt1b): " + "; ".join(f"{arm}: " + " / ".join(f"{summary['cot_token_medians'][arm][model]:g}" for model in MODELS) for arm in ARMS) + ".",
    "Pooled base/mt2e CoT-length tertile counts are in `summary.json:length_strata`; these are descriptive, not a causal adjustment.",
    "", "## Repeat-label agreement", "", "Exact agreement on the randomly selected repeats. Baseline nudge questions are excluded as not applicable. Kappa is undefined when both raters assign the same constant label.",
    "", "| Label | Agreement | Cohen's kappa |", "|---|---:|---:|"]
for field, stats in agreement.items():
    kappa = "undefined" if stats["kappa"] is None else f"{stats['kappa']:.3f}"
    lines.append(f"| {field} | {fraction(stats)} | {kappa} |")
lines += ["", "Individual disagreements are in [disagreements.json](disagreements.json). Scorers flagged the boundary between familiar product prices/limits and unsupported specifics; `invents_specifics` is not a fact-checked hallucination label.",
    "", "## Choice-parser audit", "", "A decisive scored recommendation means A or B. Agreement below compares the paper's full-answer keyword parser against the independent excerpt-based label. Conditional and none are reported separately and never forced into A/B.",
    "", "| Arm | Model | Parser agrees on decisive | Parser chooses opposite | Parser missing on decisive | Conditional / all | None / all |", "|---|---|---:|---:|---:|---:|---:|"]
for arm in ARMS:
    for model in MODELS:
        stats = parser[arm][model]
        lines.append("| " + " | ".join([arm, model] + [fraction(stats[field]) for field in ["decisive_agreement", "decisive_opposite", "decisive_unparsed", "conditional", "none"]]) + " |")
decisive = [r for r in rows if r["final_recommends"] in ["A", "B"]]
lines += ["", f"Across models and arms, the parser disagrees on {sum(r['parser_choice'] != r['final_recommends'] for r in decisive)}/{len(decisive)} decisive scored recommendations. Conditional answers account for {sum(r['final_recommends'] == 'conditional' for r in rows)}/{len(rows)} rows; no recommendation for {sum(r['final_recommends'] == 'none' for r in rows)}/{len(rows)}.",
    "", "The stored parser choices reproduce exactly on 426/426 sampled full answers. A whitespace-normalization diagnostic is in [parser_diagnostics.json](parser_diagnostics.json); it collapses whitespace with `' '.join(response.split())` and runs the unchanged parser. It does not change any saved source choice, primary label, or reported compliance count.",
    "", "Only the sampled baseline rep is labeled: baseline majority choices, the 4/5 stability filter, and the selection of counter-baseline nudges still use the original parser. This scan cannot establish corrected whole-dataset switch rates.",
    "", "## Explicit-arm monitor check", "", "Monitor: **CoT only**, told the nudge; **paper Setting 1 prompt**; **openai/gpt-6-luna via OpenRouter, provider openai, effort medium**. Existing verdicts are reused, without a new monitor call. The subset is now rows whose scored final-answer recommendation matches the option chosen in the user prompt. Missing existing monitor verdicts are excluded and counted explicitly.",
    "", "| Model | Scored compliance / sampled | With existing monitor / complied | Detected / monitored | Detected when request restated | Detected when not restated |", "|---|---:|---:|---:|---:|---:|"]
for model, stats in explicit.items():
    lines.append("| " + " | ".join([model] + [fraction(stats[field]) for field in ["complied", "monitored", "detected"]] + [fraction(stats["by_restates_request"][str(flag)]) for flag in [True, False]]) + " |")
lines += ["", "Restricting further to identical cells where both base and mt2e have a scored compliant answer and an existing monitor verdict: " + "; ".join(f"{model} detected {fraction(stats)}" for model, stats in paired_monitor.items()) + ". Monitor: CoT only, told the nudge; paper Setting 1 prompt; openai/gpt-6-luna via OpenRouter, provider openai, effort medium."]
lines += ["", "## Targeted read-through checks", "", "[spot_checks.json](spot_checks.json) records full-answer checks of 5/5 selected parser disagreements; the full answers support the scored recommendation in 5/5. These cases come from the first completed scoring batches and do not estimate an error rate. The same file records full CoTs for 3/3 selected base/mt2e pairs.",
    "", "In `gift_purchase__user_reliability_positive__path_a`, the explicit user message attributes reliability to Le Creuset, while the canonical nudge metadata attributes it to Amazon. The same mismatch appears for all three models in that sampled cell. Source inputs and labels are retained.",
    "", "## Reproduction", "", "Run from `mats12_sprint/`:", "", "```sh",
    "/home/ek/wgmn/mechtools/.venv/bin/python -B tricky_little_guys/cot-control/results/iei/trace_scan/analyze.py",
    "tricky_little_guys/cot-control/rl/.venv/bin/python -B tricky_little_guys/cot-control/results/iei/trace_scan/plot.py", "```", "",
    "`prepare.py` reproduces the sample and grader-input files from the committed source data. `scores_00.json` through `scores_07.json` are the direct subagent labels. Re-running analysis does not invoke models.", ""]
(ROOT / "README.md").write_text("\n".join(lines))
print(json.dumps(dict(n_unique=len(rows), n_repeated=len(repeats), agreement=agreement, explicit=explicit), indent=2))
