"""The Chapter 36 Meta geo holdout, as numbers a model can be calibrated to.

Sedge Shoes pauses Meta for six weeks in eight treatment regions
(out of 18), each matched with a business-as-usual control region.
It is a holdout (a "go dark" test), not a push.
What an experiment measures depends on its design:

* A push (extra spend) measures the return on the extra pounds, which is a
  marginal quantity near the starting spend level.
  On a concave response curve it cannot exceed the marginal return at the
  start, which for Meta is 0.9x at current spend on the true curve (the
  fitted model of Chapter 37 puts it near 1.1x).
* A holdout (spend switched off) measures the return on the spend that was
  switched off, which is an average-type quantity for that slice of spend.
  On the true curve it sits between the marginal return (0.9x) and,
  roughly, the annual average (1.75x).

The numbers below are derived from the generator's true response curve
(``sedge.py``) by ``ledger_holdout_design`` and ``placebo_se`` (``ledger``
in the names means the true values built into the data), and the unit
tests recompute them.
They are rounded the way the book rounds them.

Design (all weeks are zero-indexed positions in the 156-week panel):

* Regions are ranked on pre-period Meta spend (weeks 0 to 147) and paired
  with their neighbour in the ranking, 16 regions in 8 pairs.
  One region of each pair is switched off, chosen by
  ``numpy.random.default_rng(0)``; the two lowest-spending regions are left
  out. The generator holds the paid-social plan steady from week 144, so
  the test starts from business-as-usual spend.
* Meta is paused in the treatment regions in weeks 148 to 153.
  Business-as-usual Meta spend in those regions ran at GBP 12,870 a week
  (about GBP 12,900), so GBP 77,223 (about GBP 77,000) was withheld.
* The lift is measured over the six test weeks plus a two-week cooldown,
  eight weeks in all (weeks 148 to 155).
  Adstock carries a lost week's effect forward (Meta retains 40 per cent a
  week), so the cooldown captures about 99 per cent of the carryover.
  Measuring over the six test weeks alone would miss about 6 per cent of the
  lost revenue.
* The ledger curve says the pause costs GBP 120,028 of revenue over that
  window, a ROAS of 1.55 on the withheld spend.
* The estimator is a difference in differences (treatment minus control
  scaled by the pre-period revenue ratio, summed over 8 pairs and 8 weeks)
  after removing the observed price, promotion, distribution, stock,
  weather and launch effects.
  Run on the 90 eight-week placebo windows that do not overlap a known
  intervention, its standard deviation is GBP 45,013, so the standard error
  is about GBP 45,000, or 0.58 in ROAS.
* The observed lift is GBP 138,000, about 0.4 standard errors above what
  the ledger curve predicts, which is ordinary sampling noise.
  The experiment ROAS is 138,000 / 77,000 = 1.79, with a 95 per cent
  interval of roughly 0.6 to 2.9.

The case-study model (``mmm_book.case_model``, notebook 36) adds one
likelihood term, as the book describes,

    EXPERIMENT_LIFT_GBP ~ Normal(implied_lift, EXPERIMENT_SE_GBP),

where ``implied_lift`` is the revenue the fitted Meta curve says the pause
cost: Meta switched off in the treatment regions for the six test weeks,
pushed through the model's own adstock and saturation, and summed over the
test weeks plus the cooldown.
The national lab version (lab 24, ``examples/ex_calibration.py``) uses a
simpler stand-in, ``implied_meta_roas`` times
``EXPERIMENT_WITHHELD_SPEND_GBP``; it is an approximation because the ledger
curve's holdout return (1.55) sits a little below the annual average (1.75).
"""

from __future__ import annotations

import numpy as np

EXPERIMENT_DESIGN = "holdout"  # Meta paused in the treatment regions (not a push)
EXPERIMENT_WEEKS = 6
EXPERIMENT_COOLDOWN_WEEKS = 2
EXPERIMENT_MEASURED_WEEKS = EXPERIMENT_WEEKS + EXPERIMENT_COOLDOWN_WEEKS
EXPERIMENT_TREATMENT_REGIONS = 8
EXPERIMENT_CONTROL_REGIONS = 8
EXPERIMENT_TOTAL_REGIONS = 18
EXPERIMENT_START_WEEK = 148  # zero-indexed week of the first paused week

# Business-as-usual Meta spend, treatment regions together, per week.
EXPERIMENT_BASELINE_WEEKLY_GBP = 12_900.0
# Spend switched off over the six test weeks (12,870 x 6 = 77,223, rounded).
EXPERIMENT_WITHHELD_SPEND_GBP = 77_000.0
# What the ledger response curve says the pause costs over test plus cooldown.
EXPERIMENT_LEDGER_EXPECTED_LIFT_GBP = 120_000.0
# What the team observed, and the standard error from the design.
EXPERIMENT_LIFT_GBP = 138_000.0
EXPERIMENT_SE_GBP = 45_000.0

EXPERIMENT_ROAS = EXPERIMENT_LIFT_GBP / EXPERIMENT_WITHHELD_SPEND_GBP  # 1.79
EXPERIMENT_ROAS_SE = EXPERIMENT_SE_GBP / EXPERIMENT_WITHHELD_SPEND_GBP  # about 0.58
EXPERIMENT_LEDGER_EXPECTED_ROAS = (
    EXPERIMENT_LEDGER_EXPECTED_LIFT_GBP / EXPERIMENT_WITHHELD_SPEND_GBP
)  # about 1.56

_CHANNEL = "meta"
_PAIRING_SEED = 0
_PRE_PERIOD_WEEKS = 148


def select_pairs(spend: np.ndarray, seed: int = _PAIRING_SEED) -> tuple[list[int], list[int]]:
    """Pair regions on pre-period Meta spend and randomise within each pair.

    ``spend`` is a (regions, weeks) array of Meta spend.
    Returns the treatment and control region indices (8 each); the two
    regions that rank last are not used.
    """
    order = np.argsort(-spend[:, :_PRE_PERIOD_WEEKS].mean(axis=1))
    rng = np.random.default_rng(seed)
    treatment, control = [], []
    for k in range(EXPERIMENT_TREATMENT_REGIONS):
        a, b = int(order[2 * k]), int(order[2 * k + 1])
        if rng.integers(2):
            a, b = b, a
        treatment.append(a)
        control.append(b)
    return sorted(treatment), sorted(control)


def _window_effects(spend_factor, seed: int | None = None) -> dict:
    """Revenue effect of rescaling Meta spend in the treatment regions' test weeks.

    ``spend_factor`` is a scalar, or a function of the regions' business-as-usual
    window spend that returns the factor to apply (so a push can be sized in
    pounds).  0 is a pause.  Effects are pushed through the same adstock and
    Hill saturation the generator uses and summed over the test weeks alone
    and over the test weeks plus the cooldown.
    """
    from mmm_book import RANDOM_SEED
    from mmm_book.adstock import geometric_adstock
    from mmm_book.saturation import hill_saturation
    from mmm_book.sedge import ADSTOCK_ALPHA, HILL_SLOPE, _build

    _, regions, scale = _build(156, EXPERIMENT_TOTAL_REGIONS, RANDOM_SEED if seed is None else seed)
    spend = np.array([r["channel_spend"][_CHANNEL] for r in regions])
    treatment, control = select_pairs(spend)
    w0, w1 = EXPERIMENT_START_WEEK, EXPERIMENT_START_WEEK + EXPERIMENT_WEEKS
    bau_total = float(spend[treatment, w0:w1].sum())
    factor = spend_factor(bau_total) if callable(spend_factor) else spend_factor

    changed = 0.0
    effect_test_only = 0.0
    effect_with_cooldown = 0.0
    for i in treatment:
        region = regions[i]
        bau = region["channel_spend"][_CHANNEL]
        treated = bau.copy()
        treated[w0:w1] = bau[w0:w1] * factor

        def response(s, region=region):
            adstocked = geometric_adstock(
                s, alpha=ADSTOCK_ALPHA[_CHANNEL], l_max=12, normalize=True
            )
            shape = hill_saturation(
                adstocked, half=region["channel_half"][_CHANNEL], slope=HILL_SLOPE[_CHANNEL]
            )
            return shape * scale[_CHANNEL]

        effect = response(treated) - response(bau)
        changed += float((treated - bau)[w0:w1].sum())
        effect_test_only += float(effect[w0:w1].sum())
        effect_with_cooldown += float(effect[w0 : w1 + EXPERIMENT_COOLDOWN_WEEKS].sum())

    return {
        "treatment_regions": treatment,
        "control_regions": control,
        "bau_spend": bau_total,
        "spend_change": changed,
        "effect_test_only": effect_test_only,
        "effect_with_cooldown": effect_with_cooldown,
    }


def ledger_holdout_design(seed: int | None = None) -> dict:
    """Recompute the holdout's spend and expected lift from the ledger curve.

    Switches Meta off in the treatment regions for the test weeks and sums
    the lost revenue over the test weeks plus the cooldown.
    Also returns the lift that a six-week-only window would capture.
    """
    e = _window_effects(0.0, seed)
    withheld = e["bau_spend"]
    lift = -e["effect_with_cooldown"]
    return {
        "treatment_regions": e["treatment_regions"],
        "control_regions": e["control_regions"],
        "weekly_bau_spend": withheld / EXPERIMENT_WEEKS,
        "withheld_spend": withheld,
        "expected_lift_test_only": -e["effect_test_only"],
        "expected_lift": lift,
        "expected_roas": lift / withheld,
    }


def ledger_push_design(weekly_total_gbp: float = 43_000.0, seed: int | None = None) -> dict:
    """What the superseded design would have read on the ledger curve.

    Raises Meta in the same regions to ``weekly_total_gbp`` a week in total
    (13,000 to 43,000 by default) and returns the extra spend, the extra
    revenue over the test weeks plus the cooldown, and their ratio, the
    return on the extra pounds.  On the ledger curve this is well below the
    holdout's return and below 1x, which is why a push cannot read 1.75x.
    """
    e = _window_effects(lambda bau: weekly_total_gbp * EXPERIMENT_WEEKS / bau, seed)
    return {
        "extra_spend": e["spend_change"],
        "extra_revenue": e["effect_with_cooldown"],
        "roas": e["effect_with_cooldown"] / e["spend_change"],
    }


def placebo_se() -> dict:
    """Standard error of the holdout estimator from placebo windows.

    Applies the same difference-in-differences calculation to every
    eight-week window from week 13 to the start of the test that does not
    overlap a known intervention (no experiment), after stripping the
    observed price, promotion, distribution, stock, weather and launch
    effects from revenue.
    The spread of those placebo "lifts" is the design's standard error.
    """
    from mmm_book import RANDOM_SEED, sedge
    from mmm_book.sedge import correct_distribution, generate_sedge

    df = correct_distribution(generate_sedge(seed=RANDOM_SEED))
    df["_r"] = df["region"].str.split("_").str[1].astype(int)

    def panel(col):
        return df.pivot(index="week", columns="_r", values=col).sort_index(axis=1)

    uplift = np.zeros(df["week"].max() + 1)
    for start, size in sedge.LAUNCHES:
        uplift[start : start + sedge.LAUNCH_LENGTH] = size
    observed_effects = (
        panel("distribution")
        * (panel("stock") / sedge._STOCK_MEAN)
        * (sedge.PRICE_LIST / panel("price")) ** sedge._PRICE_ELASTICITY
        * (1 + sedge._PROMO_UPLIFT_PER_DEPTH * panel("promo_depth"))
        * np.exp(
            sedge._TEMP_EFFECT * (panel("temperature") - 10.0)
            + sedge._RAIN_EFFECT * (panel("rainfall") - 15.0)
        )
        * (1 + uplift[panel("revenue").index.to_numpy()])[:, None]
    )
    adjusted = panel("revenue") / observed_effects

    meta = panel("meta_spend").to_numpy().T
    treatment, control = select_pairs(meta)
    treated = adjusted[treatment].to_numpy()[:_PRE_PERIOD_WEEKS]
    controls = adjusted[control].to_numpy()[:_PRE_PERIOD_WEEKS]

    # Windows that overlap a known intervention in these regions (the
    # search-pause test, the new retailer's opening) are not placebos.
    known = set(sedge.SEARCH_PAUSE["weeks"])
    onboarding = sedge.RETAILER_ONBOARDING["week"]
    known |= set(range(onboarding, onboarding + sedge._RETAILER_OOH_WEEKS))
    width = EXPERIMENT_MEASURED_WEEKS
    estimates = []
    for start in range(13, _PRE_PERIOD_WEEKS - width + 1):
        if known & set(range(start, start + width)):
            continue
        keep = np.ones(_PRE_PERIOD_WEEKS, dtype=bool)
        keep[:13] = False
        keep[start : start + width] = False
        ratio = treated[keep].sum(axis=0) / controls[keep].sum(axis=0)
        estimates.append(
            (treated[start : start + width] - ratio * controls[start : start + width]).sum()
        )
    estimates = np.asarray(estimates)
    return {"n_windows": len(estimates), "se": float(estimates.std(ddof=1))}
