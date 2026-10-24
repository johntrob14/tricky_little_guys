import json
import random
import subprocess
from pathlib import Path

import pytest

import obf_data
import sft_inspect as S

ARENA_PY = Path(__file__).resolve().parents[3] / "arena/.venv/bin/python"
POOL = [dict(qid="aaaaaaaa-dapo", source="dapo", question="What is 6*7?", options=None, answer=42, correct_words=200.0, correct_tokens=400.0),
        dict(qid="bbbbbbbb-gpqa", source="gpqa", question="Which is a noble gas?", options=["Iron", "Neon", "Salt", "Water"], answer="B",
             correct_words=100.0, correct_tokens=180.0)]


def cand(qid="aaaaaaaa-dapo", type="sad", **kw) -> dict:
    c = dict(qid=qid, source="dapo", type=type, value=None, frame="T1", term="reasoning", placement="developer", position=None,
             system="Be sad in your reasoning.", user="Question: What is 6*7?", instruction="Be sad.", requirement="sad remarks throughout",
             true_answer=42, origin="organism", source_ref=None, reasoning="So sad. 6*7=42. Sad.", response="ANSWER: 42", cot_tokens=[1, 2, 3],
             answer=42, correct=True, n_tokens=300, words=220, band_lo=100.0, band_hi=400.0, fmt_ok=True, strict=True, partial=1.0,
             judged=True, gate=True, gate_final="YES the work is there", req_score=10, req_final="Score: 10\n" + "x" * 500, meta=True,
             meta_final="T: no mention", prog={"markers": 5}, accepted=True, reject_reasons=[], claude_ok=None, claude_note=None)
    return c | kw


def test_rebuilt_question_matches_obf_data():
    pool = [json.loads(line) for line in (S.HERE / "data/obf_train_gptoss_r2.jsonl").open()]
    for src in ("dapo", "gpqa"):
        q = next(q for q in pool if q["source"] == src)
        assert S.rebuild_question(q) == obf_data.question_block(q)


def test_select_accepted_first_and_excludes_claude_bad():
    cs = [cand(qid=f"q{i}", accepted=i < 3, claude_ok=False if i == 0 else None) for i in range(6)]
    got = S.select(cs, 5, 0, False)["sad"]
    assert sorted(c["qid"] for c in got) == ["q1", "q2"]
    got = S.select(cs, 5, 0, True)["sad"]
    assert {c["qid"] for c in got[:2]} == {"q1", "q2"} and len(got) == 5
    assert [c["qid"] for c in S.select(cs, 0, 3, True)["sad"]] == [c["qid"] for c in S.select(cs, 0, 3, True)["sad"]]


def test_select_spreads_problems():
    cs = [cand(qid=q) for q in ["a", "a", "a", "b", "c"]]
    got = S.select(cs, 3, 1, False)["sad"]
    assert sorted(c["qid"] for c in got) == ["a", "b", "c"]


def test_score_fields_map_candidate():
    q = POOL[0]
    c = cand(force={"p_correct": 0.8, "uplift": 2.1, "cot_lp_mean": -0.4, "cot_lp": [-0.1, -0.2]},
             judge2={"ok": False, "req_score": 7, "req_final": "Score: 7"}, claude_ok=False, claude_note="bookends only")
    s = S.score_fields(c, q)
    assert s["req_score"] == dict(value=10.0, explanation=c["req_final"]) and s["strict"]["explanation"] == c["req_final"]
    assert s["gate"]["explanation"] == "YES the work is there" and s["meta"] == dict(value=1, explanation="T: no mention")
    assert s["correct"] == dict(value=1, answer="42") and s["accepted"]["value"] == 1
    assert s["judge2"]["value"] == 0 and "Score: 7" in s["judge2"]["explanation"]
    assert s["p_correct"]["value"] == 0.8 and "cot_lp:" not in s["p_correct"]["explanation"]
    assert s["len_ratio"]["value"] == pytest.approx(1.1) and s["tok_ratio"]["value"] == pytest.approx(0.75)
    assert s["claude_ok"] == dict(value=0, explanation="bookends only")
    m = S.metadata(c, q)
    assert m["placement"] == "developer" and m["force"] == {"p_correct": 0.8, "uplift": 2.1, "cot_lp_mean": -0.4} and "cot_tokens" not in m
    assert S.target(c | {"answer": 41}, q) == "42" and S.target(c | {"true_answer": None}, q) == "42"


def test_score_fields_skip_missing():
    c = {k: v for k, v in cand().items() if k not in ("gate", "meta", "accepted", "req_score", "partial", "strict")}
    assert set(S.score_fields(c, None)) == {"correct", "fmt_ok"}


def test_plain_candidates(tmp_path):
    rows = [dict(qid="aaaaaaaa-dapo", source="dapo", i=i, true_answer=42, words=180 + i, reasoning=f"r{i}", response="ANSWER: 42", answer=42,
                 tokens=[1], n_tokens=350, truncated=False, correct=i != 1) for i in range(3)]
    rows += [dict(qid="bbbbbbbb-gpqa", source="gpqa", i=0, true_answer="B", words=90, reasoning="neon", response="ANSWER: B", answer="B",
                  tokens=[1], n_tokens=170, truncated=False, correct=True),
             dict(qid="bbbbbbbb-gpqa", source="gpqa", i=0, nocot=True, true_answer="B", response="ANSWER: B", answer="B", correct=True)]
    p = tmp_path / "samples.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    got = S.plain_candidates(p, {q["qid"]: q for q in POOL}, 2, 0)
    assert sorted(c["qid"] for c in got) == ["aaaaaaaa-dapo", "bbbbbbbb-gpqa"]
    g = next(c for c in got if c["source"] == "gpqa")
    assert g["user"] == S.rebuild_question(POOL[1]) and g["reasoning"] == "neon" and g["band_hi"] == 200.0 and g["origin"] == "base"
    assert next(c for c in got if c["source"] == "dapo")["source_ref"] in ("aaaaaaaa-dapo:0", "aaaaaaaa-dapo:2")


READ_BACK = """
import json, sys
from pathlib import Path
from inspect_ai.log import read_eval_log
out = {}
for p in sorted(Path(sys.argv[1]).glob("*.eval")):
    log = read_eval_log(str(p))
    out[p.stem] = dict(task=log.eval.task, results={s.name: {k: m.value for k, m in s.metrics.items()} for s in log.results.scores},
                       samples=[dict(id=s.id, target=s.target, roles=[m.role for m in s.input],
                                     reasoning=s.messages[-1].content[0].reasoning, text=s.messages[-1].content[-1].text,
                                     scores={k: [v.value, v.explanation, v.answer] for k, v in s.scores.items()}, metadata=s.metadata)
                                for s in log.samples])
print(json.dumps(out))
"""


@pytest.mark.skipif(not ARENA_PY.exists(), reason="arena venv (inspect_ai) not present")
def test_export_round_trip(tmp_path):
    (tmp_path / "pool.jsonl").write_text("".join(json.dumps(q) + "\n" for q in POOL))
    cs = [cand(qid=f"{i:08d}-x", true_answer=42, answer=42 if i else 41, correct=bool(i), accepted=i != 2) for i in range(4)]
    cs += [cand(qid="bbbbbbbb-gpqa", source="gpqa", type="happy", placement="user", system=None, true_answer="B", answer="B",
                force={"p_correct": 0.9, "uplift": 1.0, "cot_lp_mean": -0.5})]
    random.Random(0).shuffle(cs)
    (tmp_path / "c.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cs))
    plain = [dict(qid="bbbbbbbb-gpqa", source="gpqa", i=3, true_answer="B", words=90, reasoning="neon", response="ANSWER: B", answer="B",
                  tokens=[1], n_tokens=170, truncated=False, correct=True)]
    (tmp_path / "samples.jsonl").write_text(json.dumps(plain[0]) + "\n")
    out = tmp_path / "logs"
    subprocess.run([str(ARENA_PY), str(S.HERE / "sft_inspect.py"), "--candidates", str(tmp_path / "c.jsonl"), "--out-dir", str(out),
                    "--pool", str(tmp_path / "pool.jsonl"), "--per-type", "5", "--plain-samples", str(tmp_path / "samples.jsonl"),
                    "--plain-n", "8"], check=True, capture_output=True, text=True)
    logs = json.loads(subprocess.run([str(ARENA_PY), "-c", READ_BACK, str(out)], check=True, capture_output=True, text=True).stdout)
    assert set(logs) == {"sad", "happy", "plain-gptoss"}
    sad = logs["sad"]["samples"]
    assert len(sad) == 3 and all(s["target"] == "42" and s["roles"] == ["system", "user"] for s in sad)
    wrong = next(s for s in sad if s["metadata"]["qid"] == "00000000-x")
    assert wrong["scores"]["correct"] == [0, None, "41"] and wrong["metadata"]["answer"] == 41
    assert sad[0]["scores"]["req_score"][1] == cand()["req_final"] and sad[0]["reasoning"].startswith("So sad")
    assert logs["sad"]["results"]["accepted"] == {"count": 3.0, "total": 3.0}
    h = logs["happy"]["samples"][0]
    assert h["roles"] == ["user"] and h["target"] == "B" and h["scores"]["p_correct"][0] == 0.9 and h["scores"]["len_ratio"][0] == 2.2
    p = logs["plain-gptoss"]["samples"][0]
    assert p["metadata"]["type"] == "plain-gptoss" and p["target"] == "B" and p["text"] == "ANSWER: B" and "accepted" not in p["scores"]
