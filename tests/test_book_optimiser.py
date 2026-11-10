"""The optimiser numbers printed in the teaching chapters 27 and 28.

Each test reruns the companion allocator on the ledger curves (the curves a
model would have if it had recovered the true values built into the data
exactly) and checks it lands on the figure the book prints, so the text and
the code cannot drift apart.

The Chapter 38 tests below run the same calculations on these teaching
curves as a cross-check of the allocator and the plan bounds. Chapter 38
itself prints the fitted model's versions, which ``test_book_part9.py`` pins.
"""

import pytest

from mmm_book import RANDOM_SEED
from mmm_book.optimise import (
    ResponseCurves,
    band,
    book_plan_bounds,
    book_plan_weekly,
    greedy_allocate,
    ledger_betas,
)
from mmm_book.reference import national_aggregate
from mmm_book.sedge import CHANNELS, correct_distribution, generate_sedge

WEEKS = 52


@pytest.fixture(scope="module")
def curves():
    nat = national_aggregate(correct_distribution(generate_sedge(seed=RANDOM_SEED)))
    return ResponseCurves(nat, ledger_betas(nat))


def annual_m(weekly: float) -> float:
    return weekly * WEEKS / 1e6


def revenue_m(curves, alloc, cpm_rise: float = 0.0) -> float:
    """Annual incremental revenue (GBP m); ``cpm_rise`` makes every pound buy less."""
    return annual_m(
        sum(curves.money_contribution(ch, alloc[ch] / (1 + cpm_rise)) for ch in CHANNELS)
    )


def chapter27_bounds(curves):
    """Plus or minus 20 per cent for mature channels, 30 per cent for YouTube and TV."""
    lo, hi = band(curves, 0.20)
    for ch in ("youtube", "tv"):
        lo[ch] = curves.current_money[ch] * 0.70
        hi[ch] = curves.current_money[ch] * 1.30
    return lo, hi


def targeted_cut(curves, budget_m: float):
    """A budget cut taken from the lowest marginal returns, each channel within 40 per cent."""
    lo, hi = band(curves, 0.40)
    return greedy_allocate(curves, lo, hi, total=budget_m * 1e6 / WEEKS)


def test_current_mix_delivers_the_books_12_72m(curves):
    assert revenue_m(curves, curves.current_money) == pytest.approx(12.72, abs=0.01)


def test_chapter27_unconstrained_plan(curves):
    alloc = greedy_allocate(curves)
    # Chapter 27's plans table prints the allocation to two decimals
    book = {"tv": 3.00, "search": 1.19, "meta": 0.68, "youtube": 1.13, "ooh": 0.00}
    for ch in CHANNELS:
        assert round(annual_m(alloc[ch]), 2) == book[ch], ch
    gain = revenue_m(curves, alloc) - revenue_m(curves, curves.current_money)
    assert gain == pytest.approx(0.88, abs=0.005)  # 0.8847: "about 0.88 million"
    assert 0.5 * gain == pytest.approx(0.44, abs=0.005)


def test_chapter27_constrained_plan(curves):
    lo, hi = chapter27_bounds(curves)
    alloc = greedy_allocate(curves, lo, hi)
    book = {"tv": 1.82, "search": 1.22, "meta": 1.20, "youtube": 1.04, "ooh": 0.72}
    for ch in CHANNELS:
        assert round(annual_m(alloc[ch]), 2) == book[ch], ch
    assert revenue_m(curves, alloc) == pytest.approx(13.15, abs=0.01)
    gain = revenue_m(curves, alloc) - revenue_m(curves, curves.current_money)
    assert gain == pytest.approx(0.44, abs=0.01)


def test_chapter28_cut_and_recession(curves):
    current = revenue_m(curves, curves.current_money)
    cut = targeted_cut(curves, 5.1)
    assert current - revenue_m(curves, cut) == pytest.approx(0.9, abs=0.05)
    pro_rata = {ch: curves.current_money[ch] * 5.1 / 6.0 for ch in CHANNELS}
    assert current - revenue_m(curves, pro_rata) == pytest.approx(1.4, abs=0.05)
    dearer_media = revenue_m(curves, cut) - revenue_m(curves, cut, cpm_rise=0.08)
    assert dearer_media == pytest.approx(0.7, abs=0.05)


def test_chapter38_portfolio_scenarios(curves):
    plan = book_plan_weekly(curves)
    at_plan = revenue_m(curves, plan)
    assert at_plan == pytest.approx(13.0, abs=0.05)
    down = revenue_m(curves, {ch: plan[ch] * 0.85 for ch in CHANNELS})
    up = revenue_m(curves, {ch: plan[ch] * 1.10 for ch in CHANNELS})
    assert down == pytest.approx(11.5, abs=0.05)
    assert at_plan - down == pytest.approx(1.5, abs=0.05)
    assert up == pytest.approx(13.9, abs=0.05)
    assert up - at_plan == pytest.approx(0.9, abs=0.05)
    assert at_plan - revenue_m(curves, targeted_cut(curves, 5.1)) == pytest.approx(1.2, abs=0.05)


def test_chapter38_naive_and_constrained_optimisers(curves):
    current = revenue_m(curves, curves.current_money)
    lo40, hi40 = band(curves, 0.40)
    naive = greedy_allocate(curves, lo40, hi40)
    assert revenue_m(curves, naive) - current == pytest.approx(0.60, abs=0.01)

    plan_gain = revenue_m(curves, book_plan_weekly(curves)) - current
    lo, hi = book_plan_bounds(curves)
    alloc = greedy_allocate(curves, lo, hi)
    assert annual_m(alloc["meta"]) == pytest.approx(1.20, abs=0.01)
    assert annual_m(alloc["search"]) == pytest.approx(1.35, abs=0.02)
    gain = revenue_m(curves, alloc) - current
    # "about 20k more" than the printed plan
    assert gain - plan_gain == pytest.approx(0.02, abs=0.01)


def test_chapter27_table_scores_the_chapter38_naive_plan(curves):
    """Chapter 27's table prints Chapter 38's naive plan, rescored on these curves."""
    plan_m = {"search": 1.29, "meta": 0.90, "youtube": 1.12, "tv": 1.96, "ooh": 0.73}
    scale = curves.total_budget * WEEKS / 6_000_000.0
    alloc = {ch: plan_m[ch] * 1e6 / WEEKS * scale for ch in CHANNELS}
    gain = revenue_m(curves, alloc) - revenue_m(curves, curves.current_money)
    assert gain == pytest.approx(0.60, abs=0.005)
