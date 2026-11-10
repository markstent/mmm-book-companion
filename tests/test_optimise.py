import pytest

from mmm_book.optimise import (
    ResponseCurves,
    band,
    book_plan_bounds,
    book_plan_weekly,
    greedy_allocate,
    ledger_betas,
)
from mmm_book.reference import national_aggregate
from mmm_book.sedge import CHANNELS, generate_sedge

BOOK_AVERAGE = {"tv": 2.4, "search": 2.7, "meta": 1.75, "youtube": 2.0, "ooh": 1.5}
BOOK_MARGINAL = {"tv": 2.1, "search": 1.4, "meta": 0.9, "youtube": 1.8, "ooh": 1.5}


@pytest.fixture(scope="module")
def curves():
    nat = national_aggregate(generate_sedge())
    return ResponseCurves(nat, ledger_betas(nat))


def test_ledger_curves_reproduce_book_average_roas(curves):
    for ch in CHANNELS:
        assert curves.average_return(ch) == pytest.approx(BOOK_AVERAGE[ch], rel=0.03)


def test_ledger_curves_reproduce_book_marginal_roas(curves):
    for ch in CHANNELS:
        assert curves.marginal_return(ch) == pytest.approx(BOOK_MARGINAL[ch], rel=0.10)


def test_greedy_allocate_keeps_total_and_respects_bounds(curves):
    lo, hi = band(curves, 0.2)
    alloc = greedy_allocate(curves, lo, hi)
    assert sum(alloc.values()) == pytest.approx(curves.total_budget)
    for ch in CHANNELS:
        assert lo[ch] - 1e-6 <= alloc[ch] <= hi[ch] + 1e-6


def test_book_plan_direction_and_gain(curves):
    plan = book_plan_weekly(curves)
    assert sum(plan.values()) == pytest.approx(curves.total_budget)
    assert plan["meta"] < curves.current_money["meta"]
    assert plan["tv"] > curves.current_money["tv"]
    gain = (
        curves.total_contribution(plan) - curves.total_contribution(curves.current_money)
    ) * 52
    # about 0.29m a year on the teaching curves (0.27m on the fitted model, Chapter 38)
    assert 200_000 < gain < 400_000


def test_constrained_optimiser_moves_in_the_books_direction(curves):
    lo, hi = book_plan_bounds(curves)
    alloc = greedy_allocate(curves, lo, hi)
    assert alloc["meta"] < curves.current_money["meta"]
    assert alloc["tv"] > curves.current_money["tv"]
    assert alloc["youtube"] >= curves.current_money["youtube"]
