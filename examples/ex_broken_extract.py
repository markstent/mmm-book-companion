"""Chapter 30 lab: a Search extract missing 30 per cent of its spend moves ROAS, not the model."""

from mmm_book import RANDOM_SEED
from mmm_book.optimise import ResponseCurves, ledger_betas
from mmm_book.reference import national_aggregate
from mmm_book.sedge import correct_distribution, generate_sedge

DROPPED = 0.30  # the partner-network share missing from the extract


def search_roas(dropped: float = 0.0) -> float:
    """Search's average ROAS when the extract is missing ``dropped`` of its spend.

    The model credits Search with the same revenue either way, so only the
    spend in the denominator changes.
    """
    nat = national_aggregate(correct_distribution(generate_sedge(seed=RANDOM_SEED)))
    curves = ResponseCurves(nat, ledger_betas(nat))
    revenue = curves.contribution("search", curves.current_native["search"]) * len(nat)
    spend = nat["search_spend"].sum() * (1 - dropped)
    return float(revenue / spend)


def main() -> dict:
    before = search_roas()
    after = search_roas(DROPPED)
    print(f"Search ROAS, complete extract:   {before:.2f}")
    print(f"Search ROAS, 30% of spend gone:  {after:.2f}")
    print(f"rise: {after / before - 1:.0%}  (1 / 0.7 = {1 / (1 - DROPPED):.2f})")
    return {"before": before, "after": after, "rise": after / before - 1}


if __name__ == "__main__":
    main()
