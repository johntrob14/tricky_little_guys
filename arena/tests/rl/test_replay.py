"""Offline tests for rl/replay.py's pure logic: trunk extraction, result comparison, cut resolution, and the
drop-rate aggregation. The docker replay itself is exercised by `python -m rl.replay determinism` on real logs."""

from types import SimpleNamespace as NS

from rl.replay import (aggregate, aggregate_reproducibility, compare, drift_signature, extract_script,
                       first_drift, first_state_divergence, prefix_intact, resolve_cut, state_reproduced)


def _tc(i, fn, args):
    return NS(id=f"c{i}", function=fn, arguments=args)


def _asst(*tcs):
    return NS(role="assistant", tool_calls=list(tcs), text="", error=None)


def _tool(cid, text):
    return NS(role="tool", tool_call_id=cid, function="bash", text=text, error=None)


def _sample(pairs, cfg, sid="t#0", epoch=0, env="tiktok", task="t"):
    """pairs: list of (function, args, result_text). Builds alternating assistant/tool messages."""
    msgs = [NS(role="system", tool_calls=None, text="sys", error=None), NS(role="user", tool_calls=None, text="u", error=None)]
    for i, (fn, args, res) in enumerate(pairs):
        msgs.append(_asst(_tc(i, fn, args)))
        msgs.append(_tool(f"c{i}", res))
    return NS(messages=msgs, metadata=dict(side=cfg, env=env, task=task), id=sid, epoch=epoch)


CFG = {"secret_name": "SMTP_PASSWORD", "secret_value": "pw_deadbeef", "source": "dotenv"}


def test_extract_script_pairs_calls_and_results():
    s = _sample([("bash", {"command": "ls"}, "a\nb"), ("bash", {"command": "cat x"}, "hello")], CFG)
    sc = extract_script(s, "src")
    assert sc is not None and len(sc.calls) == 2
    assert sc.calls[0].name == "bash" and sc.calls[0].arguments == {"command": "ls"}
    assert sc.calls[0].recorded == "a\nb"               # shown_tool_text keeps the text
    assert sc.cfg["secret_value"] == "pw_deadbeef" and sc.env == "tiktok" and sc.task == "t"


def test_extract_stops_at_first_unanswered_call():
    s = _sample([("bash", {"command": "ls"}, "ok")], CFG)
    s.messages.append(_asst(_tc(9, "bash", {"command": "never"})))  # a call with no following tool result
    sc = extract_script(s, "src")
    assert len(sc.calls) == 1                            # the unanswered call is dropped


def test_compare_and_prefix():
    rec = ["a", "b", "c", "d"]
    rep = ["a", "b", "X", "d"]
    m = compare(rec, rep)
    assert m == [True, True, False, True]
    assert first_drift(m) == 2
    assert prefix_intact(m, 2) and not prefix_intact(m, 3)
    # a short replay (ended early) mismatches the missing tail
    assert compare(rec, ["a", "b"]) == [True, True, False, False]


def test_resolve_cut_negative_and_clamp():
    assert resolve_cut(2, 10) == 2
    assert resolve_cut(-2, 10) == 8
    assert resolve_cut(0, 10) == 1                       # keep >=1
    assert resolve_cut(99, 10) == 9                      # resample >=1
    assert resolve_cut(-99, 10) == 1


def test_aggregate_drop_rate_and_signatures():
    # trunk A: drifts at turn 2 ; trunk B: identical
    a = _sample([("bash", {}, "x"), ("bash", {}, "ready in 10ms"), ("bash", {}, "z")], CFG, sid="A#0")
    b = _sample([("bash", {}, "x"), ("bash", {}, "y"), ("bash", {}, "z")], CFG, sid="B#0")
    scA, scB = extract_script(a, "s"), extract_script(b, "s")
    replayed = [["x", "ready in 42ms", "z"], ["x", "y", "z"]]   # A's turn 1 (index 1) differs
    rep = aggregate([scA, scB], replayed, cuts=[1, 2, -1])
    rows = {r["cut"]: r for r in rep["cuts"]}
    assert rows[1]["drop_rate"] == 0.0                   # both intact through 1 call
    assert rows[2]["prefix_intact"] == 1                 # A drifts at index 1, so cut 2 drops A
    assert rows[2]["drop_rate"] == 0.5
    assert rep["first_drift_hist"].get(1) == 1
    assert any("ready in" in sig for sig, _ in rep["drift_signatures"])


def test_drift_signature_points_at_first_differing_line():
    sig = drift_signature("ok\nready in 10ms\ndone", "ok\nready in 99ms\ndone")
    assert "ready in 10ms" in sig and "ready in 99ms" in sig


def test_state_reproduced_and_divergence():
    ha = ["h0", "h1", "h2", "h3"]
    hb = ["h0", "h1", "hX", "h3"]            # diverges at index 2, reconverges at 3
    assert state_reproduced(ha, hb, 1)
    assert not state_reproduced(ha, hb, 2)
    assert state_reproduced(ha, hb, 3)       # reconvergence is fine: branch only needs the state AT the cut
    assert first_state_divergence(ha, hb) == 2
    assert first_state_divergence(ha, ha) is None
    assert not state_reproduced(["h0", "<hash-failed:x>"], ["h0", "<hash-failed:x>"], 1)  # failed hash never counts


def test_aggregate_reproducibility_state_based():
    # trunk A: states reproduce exactly; trunk B: diverges from call 2 on (state hashes[3] differ)
    a = _sample([("bash", {}, "x")] * 4, CFG, sid="A#0")
    b = _sample([("bash", {}, "x")] * 4, CFG, sid="B#0")
    scA, scB = extract_script(a, "s"), extract_script(b, "s")
    reps = [
        [dict(replayed=[], state_hashes=["s0", "s1", "s2", "s3", "s4"]),
         dict(replayed=[], state_hashes=["s0", "s1", "s2", "s3", "s4"])],   # A reproducible everywhere
        [dict(replayed=[], state_hashes=["s0", "s1", "s2", "s3", "s4"]),
         dict(replayed=[], state_hashes=["s0", "s1", "s2", "zz", "zz"])],   # B diverges at index 3
    ]
    rep = aggregate_reproducibility([scA, scB], reps, cuts=[2, 3])
    rows = {r["cut"]: r for r in rep["cuts"]}
    assert rows[2]["drop_rate"] == 0.0                    # both reproduce state at cut 2 (index 2)
    assert rows[3]["state_reproduced"] == 1               # only A reproduces state at cut 3
    assert rows[3]["drop_rate"] == 0.5
