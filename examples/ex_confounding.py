"""Chapter 5 lab: a promotion riding on television, and conditioning on a website visit."""

import numpy as np

from mmm_book import RANDOM_SEED


def visitors_table() -> dict:
    """Build the chapter's 10,000 people (Table 5.2) and count who visits and buys.

    Half of each intent group is advertised. Visiting depends on intent and
    advertising; buying depends on intent, plus two points for advertising.
    Each cell is filled to its exact share, so the counts match the table.
    """
    people = []  # (high intent, advertised, visited, bought)
    for high in (True, False):
        for ad in (True, False):
            n = 1_000 if high else 4_000
            visits = round(n * (0.8 if high else 0.4 if ad else 0.1))
            buys = round(visits * ((0.30 if high else 0.05) + 0.02 * ad))
            for i in range(n):
                people.append((high, ad, i < visits, i < buys))
    high, ad, visit, buy = np.array(people).T

    def rate(mask):
        return float(buy[mask].mean())

    return {
        "visitors_ad": rate(ad & visit),
        "visitors_no_ad": rate(~ad & visit),
        "high_visitors_ad": rate(ad & visit & high),
        "high_visitors_no_ad": rate(~ad & visit & high),
        "low_visitors_ad": rate(ad & visit & ~high),
        "low_visitors_no_ad": rate(~ad & visit & ~high),
    }


def promotion_weeks(
    n_weeks: int = 156,
    tv_effect: float = 85_000.0,
    promo_effect: float = 150_000.0,
    seed: int = RANDOM_SEED,
) -> dict:
    """Weeks where a 20 per cent promotion is launched with television, as in Table 5.1."""
    rng = np.random.default_rng(seed)
    tv = (rng.random(n_weeks) < 0.3).astype(float)
    promo = np.where(tv == 1, rng.random(n_weeks) < 0.95, rng.random(n_weeks) < 0.10)
    promo = promo.astype(float)
    revenue = 820_000 + tv_effect * tv + promo_effect * promo + rng.normal(0, 15_000, n_weeks)
    ones = np.ones(n_weeks)
    naive = np.linalg.lstsq(np.column_stack([ones, tv]), revenue, rcond=None)[0]
    adjusted = np.linalg.lstsq(np.column_stack([ones, tv, promo]), revenue, rcond=None)[0]
    return {
        "naive_tv": float(naive[1]),
        "adjusted_tv": float(adjusted[1]),
        "adjusted_promo": float(adjusted[2]),
    }


def main() -> dict:
    v = visitors_table()
    print("Conversion, advertised vs not advertised, among visitors")
    print(f"  all visitors:       {v['visitors_ad']:.1%} vs {v['visitors_no_ad']:.1%}  (Table 5.2)")
    print(f"  high intent only:   {v['high_visitors_ad']:.1%} vs {v['high_visitors_no_ad']:.1%}")
    print(f"  low intent only:    {v['low_visitors_ad']:.1%} vs {v['low_visitors_no_ad']:.1%}")
    w = promotion_weeks()
    print("\nTelevision's weekly effect (true value 85,000; the promotion adds 150,000)")
    print(f"  promotion left out: {w['naive_tv']:,.0f}")
    print(f"  promotion included: {w['adjusted_tv']:,.0f}  (promotion {w['adjusted_promo']:,.0f})")
    return {**v, **w}


if __name__ == "__main__":
    main()
