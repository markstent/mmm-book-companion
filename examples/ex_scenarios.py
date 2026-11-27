"""Chapter 25 lab: four ways to cost a 15 per cent budget cut, on the teaching curves."""

from mmm_book import RANDOM_SEED
from mmm_book.optimise import ResponseCurves, band, greedy_allocate, ledger_betas
from mmm_book.reference import national_aggregate
from mmm_book.sedge import CHANNELS, correct_distribution, generate_sedge

WEEKS = 52
MARGIN = 0.5  # Sedge's margin on incremental revenue
CUT = 0.15


def teaching_curves() -> ResponseCurves:
    nat = national_aggregate(correct_distribution(generate_sedge(seed=RANDOM_SEED)))
    return ResponseCurves(nat, ledger_betas(nat))


def annual_revenue(curves: ResponseCurves, plan: dict[str, float]) -> float:
    """Incremental revenue a year (GBP) of a weekly plan."""
    return curves.total_contribution(plan) * WEEKS


def scenarios(curves: ResponseCurves, cut: float = CUT, reach: float = 0.40) -> dict[str, dict]:
    """Revenue and profit change (GBP m a year) of each costing of the cut."""
    now = curves.current_money
    total = curves.total_budget
    current = annual_revenue(curves, now)
    lo, hi = band(curves, reach)
    no_rise = {ch: now[ch] for ch in CHANNELS}
    plans = {
        "equal cut": {ch: now[ch] * (1 - cut) for ch in CHANNELS},
        f"targeted, within +/-{reach:.0%}": greedy_allocate(
            curves, lo, hi, total=total * (1 - cut)
        ),
        "targeted, no channel rises": greedy_allocate(curves, lo, no_rise, total=total * (1 - cut)),
    }
    spend_saved = cut * total * WEEKS
    average_roas = current / (total * WEEKS)
    revenue = {"average-ROAS shortcut": -spend_saved * average_roas}
    revenue.update({name: annual_revenue(curves, plan) - current for name, plan in plans.items()})
    return {
        name: {"revenue": r / 1e6, "profit": (MARGIN * r + spend_saved) / 1e6}
        for name, r in revenue.items()
    }


def main() -> dict[str, dict]:
    result = scenarios(teaching_curves())
    print(f"{'costing of the 15% cut':<34}{'revenue':>10}{'profit':>10}  (GBP m a year)")
    for name, row in result.items():
        print(f"{name:<34}{row['revenue']:>+10.2f}{row['profit']:>+10.2f}")
    return result


if __name__ == "__main__":
    main()
