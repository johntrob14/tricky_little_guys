import pytest

from rl.budget import Budget, BudgetExceeded, Ledger, episode_worst_case_usd, sample_cost_usd, train_cost_usd


def test_costs_match_price_table():
    m = "openai/gpt-oss-120b"
    assert sample_cost_usd(m, 10000, 4000, 1000) == pytest.approx((6000 * 0.33 + 4000 * 0.066 + 1000 * 0.84) / 1e6)
    assert train_cost_usd(m, 20000) == pytest.approx(20000 * 0.737 / 1e6)
    assert train_cost_usd("openai/gpt-oss-120b:peft:131072", 1e6) == pytest.approx(0.737)


def test_worst_case_is_an_upper_bound_and_monotone():
    m = "openai/gpt-oss-120b"
    a = episode_worst_case_usd(m, max_turns=20, max_tokens=4096)
    assert a > sample_cost_usd(m, 15000, 0, 20 * 400) > 0
    assert episode_worst_case_usd(m, max_turns=40, max_tokens=4096) > a


def test_reserve_commit_release_and_cap(tmp_path):
    led = Ledger(tmp_path / "l.csv")
    b = Budget(1.0, stage="s", run="r", ledger=led)
    b.reserve(0.6)
    with pytest.raises(BudgetExceeded):
        b.reserve(0.5)  # reserved 0.6 + 0.5 passes 1.0
    b.release(0.6)
    b.reserve(0.5)
    b.commit(0.3, reserved_usd=0.5, kind="sample")
    assert b.committed == pytest.approx(0.3) and b.reserved == pytest.approx(0)
    with pytest.raises(BudgetExceeded):
        b.require(0.8)
    b.require(0.7)
    with pytest.raises(BudgetExceeded):
        b.commit(0.8, kind="train")  # spent is kept, and it raises
    assert b.committed == pytest.approx(1.1)
    rows = led.rows()
    assert [r["kind"] for r in rows] == ["counted", "counted"] and led.total_counted() == pytest.approx(1.1)


def test_stage_refuses_over_cap_and_over_total(tmp_path):
    led = Ledger(tmp_path / "l.csv")
    with pytest.raises(BudgetExceeded, match="stage cap"):
        Budget(1.0, stage="s", ledger=led).start_stage(1.5)
    assert led.rows() == []
    led.append("old", "r0", "m", "counted", counted_usd=19.5)
    with pytest.raises(BudgetExceeded, match="total cap"):
        Budget(3.0, stage="s", ledger=led).start_stage(1.0)
    b = Budget(3.0, stage="s", ledger=led)
    b.start_stage(0.4, note="ok")
    assert led.rows()[-1]["kind"] == "forecast" and led.rows()[-1]["estimate_usd"] == "0.400000"


def test_ledger_append_only_header_once(tmp_path):
    led = Ledger(tmp_path / "d" / "l.csv")
    for i in range(3):
        Ledger(led.path).append("p", "r", "m", "counted", counted_usd=0.1 * i)
    lines = led.path.read_text().splitlines()
    assert lines[0].startswith("time,phase,run") and len(lines) == 4
