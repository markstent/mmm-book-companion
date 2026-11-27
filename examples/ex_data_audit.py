"""Chapter 11 lab: run the audits on a Sedge extract with planted faults."""

import numpy as np
import pandas as pd

from mmm_book.sedge import generate_sedge


def audit(df: pd.DataFrame) -> dict:
    """Run the audits and return what each found; an empty entry means it found nothing."""
    week = df.groupby("week")
    found = {}

    # Completeness: a channel that is zero in every region for a week (never a true zero here)
    meta = week["meta_spend"].sum()
    found["meta_zero_weeks"] = [int(w) for w in meta.index[meta == 0]]

    # Units: OOH impressions per pound jump a thousandfold when the unit changes
    per_pound = week["ooh_impr"].sum() / week["ooh_spend"].sum()
    found["ooh_unit_weeks"] = [
        int(w) for w in per_pound.index[per_pound > 100 * per_pound.median()]
    ]

    # Structural break: the step in generic Search cost per impression that best splits the history
    cpm = (
        ((df["search_spend"] - df["search_brand_spend"]) / df["search_generic_impr"])
        .groupby(df["week"])
        .median()
    )
    best = None
    for w in range(20, len(cpm) - 20):
        before, after = cpm.iloc[:w], cpm.iloc[w:]
        t = (after.mean() - before.mean()) / np.sqrt(
            before.var() / len(before) + after.var() / len(after)
        )
        if best is None or abs(t) > abs(best[0]):
            best = (float(t), int(cpm.index[w]), float(after.mean() / before.mean() - 1))
    t_stat, break_week, step = best
    found["search_break"] = (break_week, step) if abs(t_stat) > 6 else None

    # Plausibility: a price cut with no promotion behind it (a retailer-funded discount)
    usual = df.groupby("region")["price"].transform("median")
    unpromoted = df["promo_depth"] == 0
    cut = (df["price"] / usual - 1).where(unpromoted).groupby(df["week"]).mean()
    found["unlogged_discount_weeks"] = [int(w) for w in cut.index[cut < -0.08]]

    # Reconciliation across regions: national television copied into every region
    spread = week["tv_grps"].std()
    found["tv_identical_in_every_region"] = bool((spread < 1e-9).all())
    return found


def main() -> dict:
    clean = audit(generate_sedge())
    planted = audit(generate_sedge(faults=True))
    print("exception log (faults=True)")
    print(f"  Meta zero-filled in every region, week(s): {planted['meta_zero_weeks']}")
    print(f"  OOH units changed, week(s): {planted['ooh_unit_weeks']}")
    week, step = planted["search_break"]
    print(f"  Paid Search cost definition stepped {step:.1%} from week {week}")
    print(f"  Price cut with no promotion, week(s): {planted['unlogged_discount_weeks']}")
    print(f"  Television identical in every region: {planted['tv_identical_in_every_region']}")
    print(f"\nsame audits on the clean extract: {clean}")
    return planted


if __name__ == "__main__":
    main()
