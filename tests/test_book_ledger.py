"""Tie the synthetic generator to the book's canonical Sedge Shoes ledger.

The values below are the figures printed in the book (Chapters 34, 36 and
37) and held in check_numbers.py at the repository root (CANONICAL). If
the book ledger changes, these constants and ``GROUND_TRUTH`` must change
together.
"""

import numpy as np
import pytest

from mmm_book.sedge import (
    BOOK_N_REGIONS,
    CHANNELS,
    GROUND_TRUTH,
    TV_COST_PER_GRP,
    generate_sedge,
    ground_truth_contributions,
    true_marginal_roas,
)

BOOK_AVERAGE_ROAS = {"tv": 2.4, "search": 2.7, "meta": 1.75, "youtube": 2.0, "ooh": 1.5}
BOOK_MARGINAL_ROAS = {"tv": 2.1, "search": 1.4, "meta": 0.9, "youtube": 1.8, "ooh": 1.5}
BOOK_ANNUAL_SPEND_M = {"tv": 1.4, "search": 1.4, "meta": 1.5, "youtube": 0.8, "ooh": 0.9}
BOOK_TOTAL_INCREMENTAL_M = 12.72
BOOK_ANNUAL_REVENUE_M = 48.0


@pytest.fixture(scope="module")
def panel():
    return generate_sedge()


def test_ground_truth_matches_book_ledger():
    assert GROUND_TRUTH == BOOK_AVERAGE_ROAS


def test_default_panel_has_book_dimensions(panel):
    assert BOOK_N_REGIONS == 18
    assert panel["region"].nunique() == 18
    assert len(panel) == 2808  # 156 weeks x 18 regions, Chapter 34


def _annual_spend_m(panel):
    spend = {ch: panel[f"{ch}_spend"].sum() for ch in CHANNELS if ch != "tv"}
    spend["tv"] = panel["tv_grps"].sum() * TV_COST_PER_GRP
    return {ch: v / 3.0 / 1e6 for ch, v in spend.items()}


def test_annual_media_spend_matches_book(panel):
    annual = _annual_spend_m(panel)
    for ch, target in BOOK_ANNUAL_SPEND_M.items():
        assert annual[ch] == pytest.approx(target, rel=0.05), ch
    assert sum(annual.values()) == pytest.approx(6.0, rel=0.03)


def test_annual_revenue_matches_book(panel):
    annual = panel["revenue"].sum() / 3.0 / 1e6
    assert annual == pytest.approx(BOOK_ANNUAL_REVENUE_M, rel=0.05)


def test_realised_average_roas_matches_book(panel):
    contrib = ground_truth_contributions()
    annual = _annual_spend_m(panel)
    for ch, target in BOOK_AVERAGE_ROAS.items():
        realised = contrib[ch].sum() / 3.0 / 1e6 / annual[ch]
        assert realised == pytest.approx(target, rel=1e-6), ch


def test_total_incremental_revenue_matches_book():
    contrib = ground_truth_contributions()
    total = sum(v.sum() for v in contrib.values()) / 3.0 / 1e6
    # The book's 12.72m uses its own spend figures; realised spend is
    # within a few per cent of them.
    assert total == pytest.approx(BOOK_TOTAL_INCREMENTAL_M, rel=0.05)


def test_marginal_roas_matches_book_ledger():
    marginal = true_marginal_roas()
    for ch, target in BOOK_MARGINAL_ROAS.items():
        assert marginal[ch] == pytest.approx(target, rel=0.05), ch
    # The ranking story of Chapter 37: Search leads on average but not at
    # the margin, Television leads at the margin, Meta is below 1.
    assert marginal["tv"] > marginal["youtube"] > marginal["search"]
    assert marginal["meta"] < 1.0


def test_ground_truth_values_are_finite():
    assert all(np.isfinite(v) for v in GROUND_TRUTH.values())


def test_meta_experiment_matches_chapter_36():
    from mmm_book.experiment import (
        EXPERIMENT_BASELINE_WEEKLY_GBP,
        EXPERIMENT_COOLDOWN_WEEKS,
        EXPERIMENT_DESIGN,
        EXPERIMENT_LEDGER_EXPECTED_LIFT_GBP,
        EXPERIMENT_LEDGER_EXPECTED_ROAS,
        EXPERIMENT_LIFT_GBP,
        EXPERIMENT_ROAS,
        EXPERIMENT_ROAS_SE,
        EXPERIMENT_SE_GBP,
        EXPERIMENT_WEEKS,
        EXPERIMENT_WITHHELD_SPEND_GBP,
    )

    assert EXPERIMENT_DESIGN == "holdout"
    assert EXPERIMENT_LIFT_GBP == 138_000
    assert EXPERIMENT_SE_GBP == 45_000
    assert EXPERIMENT_WITHHELD_SPEND_GBP == 77_000
    assert EXPERIMENT_COOLDOWN_WEEKS == 2
    assert EXPERIMENT_ROAS == pytest.approx(1.79, abs=0.005)
    assert EXPERIMENT_ROAS_SE == pytest.approx(0.58, abs=0.005)
    # about 12,900 a week for six weeks is the 77,000 withheld (rounded)
    assert EXPERIMENT_BASELINE_WEEKLY_GBP * EXPERIMENT_WEEKS == pytest.approx(
        EXPERIMENT_WITHHELD_SPEND_GBP, rel=0.01
    )
    # eight regions of 18 carry roughly 8/18 of the 1.5m Meta ledger
    assert EXPERIMENT_BASELINE_WEEKLY_GBP * 52 == pytest.approx(1.5e6 * 8 / 18, rel=0.03)
    # a holdout reads an average-type return: above the 0.9x marginal, near the 1.75x average
    assert EXPERIMENT_LEDGER_EXPECTED_ROAS == pytest.approx(
        EXPERIMENT_LEDGER_EXPECTED_LIFT_GBP / EXPERIMENT_WITHHELD_SPEND_GBP
    )
    marginal = true_marginal_roas()["meta"]
    assert marginal < EXPERIMENT_LEDGER_EXPECTED_ROAS < GROUND_TRUTH["meta"]
    # the observed lift is within one standard error of what the ledger curve predicts
    assert abs(EXPERIMENT_LIFT_GBP - EXPERIMENT_LEDGER_EXPECTED_LIFT_GBP) < EXPERIMENT_SE_GBP


def test_meta_holdout_numbers_follow_from_the_ledger_curve():
    from mmm_book.experiment import (
        EXPERIMENT_BASELINE_WEEKLY_GBP,
        EXPERIMENT_LEDGER_EXPECTED_LIFT_GBP,
        EXPERIMENT_SE_GBP,
        EXPERIMENT_WITHHELD_SPEND_GBP,
        ledger_holdout_design,
        placebo_se,
    )

    design = ledger_holdout_design()
    assert len(design["treatment_regions"]) == 8
    assert not set(design["treatment_regions"]) & set(design["control_regions"])
    assert design["withheld_spend"] == pytest.approx(EXPERIMENT_WITHHELD_SPEND_GBP, rel=0.005)
    assert design["weekly_bau_spend"] == pytest.approx(EXPERIMENT_BASELINE_WEEKLY_GBP, rel=0.005)
    assert design["expected_lift"] == pytest.approx(EXPERIMENT_LEDGER_EXPECTED_LIFT_GBP, rel=0.005)
    # the cooldown matters: six test weeks alone miss about 6 per cent of the lost revenue
    assert design["expected_lift_test_only"] < design["expected_lift"]
    assert design["expected_lift_test_only"] / design["expected_lift"] == pytest.approx(
        0.94, abs=0.01
    )
    assert placebo_se()["se"] == pytest.approx(EXPERIMENT_SE_GBP, rel=0.02)
    # Appendix F's checklist prints these to the pound
    assert round(design["weekly_bau_spend"]) == 12_870
    assert round(design["withheld_spend"]) == 77_223
    placebo = placebo_se()
    assert placebo["n_windows"] == 90
    assert round(placebo["se"]) == 45_013
    # "the pause cost about 120,000", 0.4 standard errors below the observed lift
    assert round(design["expected_lift"], -3) == 120_000
    assert round((138_000 - design["expected_lift"]) / EXPERIMENT_SE_GBP, 1) == 0.4


def test_a_push_could_not_have_read_the_holdout_roas():
    """The superseded design (a push from 13,000 to 43,000 a week) cannot reach 1.75x.

    On a concave curve the return on extra spend is capped by the marginal return
    at the starting point, so a push in the same eight regions reads far below
    the holdout.  That is why the Chapter 36 experiment is a holdout.
    """
    from mmm_book.experiment import (
        EXPERIMENT_LEDGER_EXPECTED_ROAS,
        ledger_push_design,
    )

    push = ledger_push_design()
    assert push["extra_spend"] == pytest.approx(180_000, rel=0.02)
    assert push["roas"] < 1.0
    assert push["roas"] < EXPERIMENT_LEDGER_EXPECTED_ROAS
    assert push["roas"] < GROUND_TRUTH["meta"]
