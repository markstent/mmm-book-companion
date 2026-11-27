"""Each new lab script runs and prints the number its chapter quotes."""

import importlib.util
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_power_is_about_half(capsys):
    """Chapter 2 and Primer 2: a power of about 52 per cent."""
    power = load("ex_power").main()
    assert power == pytest.approx(0.516, abs=0.01)
    ex = load("ex_power")
    assert ex.simulate_power(weeks=8) == pytest.approx(0.807, abs=0.01)  # the 80 per cent design


def test_confounding_reproduces_table_5_2_and_the_adjustment(capsys):
    result = load("ex_confounding").main()
    assert result["visitors_ad"] == pytest.approx(368 / 2400)  # 15.3 per cent
    assert result["visitors_no_ad"] == pytest.approx(260 / 1200)  # 21.7 per cent
    assert result["high_visitors_ad"] - result["high_visitors_no_ad"] == pytest.approx(0.02)
    assert result["low_visitors_ad"] - result["low_visitors_no_ad"] == pytest.approx(0.02)
    assert result["adjusted_tv"] == pytest.approx(85_000, abs=10_000)
    assert result["naive_tv"] > result["adjusted_tv"] + 100_000


def test_data_audit_finds_every_planted_fault(capsys):
    found = load("ex_data_audit").main()
    assert found["meta_zero_weeks"] == [75]
    assert found["ooh_unit_weeks"] == list(range(100, 108))
    assert found["search_break"][0] == 60
    assert found["search_break"][1] == pytest.approx(0.08, abs=0.02)
    assert found["unlogged_discount_weeks"] == list(range(40, 44))
    assert found["tv_identical_in_every_region"] is True


def test_data_audit_is_quiet_on_the_clean_extract():
    from mmm_book.sedge import generate_sedge

    found = load("ex_data_audit").audit(generate_sedge())
    assert found["meta_zero_weeks"] == found["ooh_unit_weeks"] == []
    assert found["search_break"] is None
    assert found["unlogged_discount_weeks"] == []
    assert found["tv_identical_in_every_region"] is False


def test_scenarios_match_chapter_25(capsys):
    result = load("ex_scenarios").main()
    assert result["average-ROAS shortcut"]["revenue"] == pytest.approx(-1.9, abs=0.02)
    assert result["average-ROAS shortcut"]["profit"] == pytest.approx(-0.05, abs=0.01)
    assert result["equal cut"]["revenue"] == pytest.approx(-1.41, abs=0.02)
    assert result["equal cut"]["profit"] == pytest.approx(0.19, abs=0.01)
    targeted = result["targeted, within +/-40%"]
    assert targeted["revenue"] == pytest.approx(-0.92, abs=0.01)
    assert targeted["profit"] == pytest.approx(0.44, abs=0.01)
    assert result["targeted, no channel rises"]["revenue"] == pytest.approx(-1.1, abs=0.02)
    assert result["targeted, no channel rises"]["profit"] == pytest.approx(0.34, abs=0.01)


def test_drift_rule_fires_on_a_shift_and_not_on_steady_weeks(capsys):
    result = load("ex_drift").main()
    assert result["steady"] is None
    assert 13 <= result["shifted"] <= 20
    assert result["false_alarm_rate"] < 0.005


def test_broken_extract_raises_search_roas_by_43_per_cent(capsys):
    result = load("ex_broken_extract").main()
    assert result["before"] == pytest.approx(2.7, abs=0.01)
    assert result["rise"] == pytest.approx(1 / 0.7 - 1, abs=0.005)


def test_readout_quotes_the_chapter_34_plan(capsys):
    text = load("ex_readout").main()
    assert "266k revenue a year" in text
    assert "| Meta | 1.50 | 1.25 |" in text
    assert "Meta, YouTube, Search, OOH, Television." in text
