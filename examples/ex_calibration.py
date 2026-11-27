"""Calibrate Meta to the Chapter 32 geo holdout (GBP 138k lift from GBP 77k of withheld spend).

The likelihood compares the measured lift with the lift the model predicts
for the test's own weeks and spend change: Meta spend in the treatment
regions removed for the six paused weeks, pushed through the same adstock
and saturation as the model's feature, and summed over the test weeks plus
the two-week cooldown (Chapter 22's formula, on the national aggregate).
"""

# [book:calibration_imports-start]
import pymc as pm

from mmm_book import RANDOM_SEED
from mmm_book.adstock import geometric_adstock
from mmm_book.experiment import (
    EXPERIMENT_LIFT_GBP,
    EXPERIMENT_MEASURED_WEEKS,
    EXPERIMENT_ROAS,
    EXPERIMENT_SE_GBP,
    EXPERIMENT_START_WEEK,
    EXPERIMENT_WEEKS,
    select_pairs,
)
from mmm_book.reference import (
    REVENUE_SCALE,
    national_aggregate,
    reference_model,
)
from mmm_book.saturation import hill_saturation
from mmm_book.sedge import (
    ADSTOCK_ALPHA,
    GROUND_TRUTH,
    HALF_MULT,
    generate_sedge,
)

# [book:calibration_imports-end]

# [book:calibration-start]
panel = generate_sedge()
nat = national_aggregate(panel)
spend = nat["meta_spend"].to_numpy(dtype=float)
alpha = ADSTOCK_ALPHA["meta"]
stock = geometric_adstock(spend, alpha)
half = HALF_MULT["meta"] * stock.mean()


def curve(s):
    """Meta's adstocked, saturated feature."""
    return hill_saturation(geometric_adstock(s, alpha), half)


# The test path: Meta off in the treatment regions
# for the six paused weeks, measured over eight.
by_region = panel.pivot(
    index="region",
    columns="week",
    values="meta_spend",
)
regional = by_region.to_numpy()
treated, _ = select_pairs(regional)
w0 = EXPERIMENT_START_WEEK
paused = spend.copy()
weeks = slice(w0, w0 + EXPERIMENT_WEEKS)
paused[weeks] -= regional[treated, weeks].sum(axis=0)
window = slice(w0, w0 + EXPERIMENT_MEASURED_WEEKS)
bau = curve(spend)
lost = float((bau - curve(paused))[window].sum())
withheld = float((spend - paused).sum())

model = reference_model(nat)
with model:
    # beta is Meta's average weekly contribution
    unit = model["beta_meta"] * REVENUE_SCALE / bau.mean()
    annual = unit * bau.sum() / spend.sum()
    roas = pm.Deterministic("meta_roas", annual)
    lift = pm.Deterministic("predicted_lift", unit * lost)
    fits = {
        "before": pm.sample(
            draws=200,
            tune=200,
            chains=2,
            cores=1,
            target_accept=0.9,
            random_seed=RANDOM_SEED,
            progressbar=False,
        )
    }
    # The measured lift is a noisy reading of the
    # predicted one, both in pounds.
    pm.Normal(
        "experiment",
        mu=lift,
        sigma=EXPERIMENT_SE_GBP,
        observed=EXPERIMENT_LIFT_GBP,
    )
    fits["after"] = pm.sample(
        draws=200,
        tune=200,
        chains=2,
        cores=1,
        target_accept=0.9,
        random_seed=RANDOM_SEED,
        progressbar=False,
    )

head = f"holdout: lift {EXPERIMENT_LIFT_GBP:,.0f}"
tail = f"({EXPERIMENT_ROAS:.2f})"
print(f"{head} on {withheld:,.0f} withheld {tail}")
for name, idata in fits.items():
    post = idata.posterior
    print(
        f"{name:6}  predicted lift "
        f"{float(post['predicted_lift'].mean()):,.0f}"
        f"  Meta ROAS {float(post['meta_roas'].mean()):.2f}"
    )
print(f"planning value (truth): {GROUND_TRUTH['meta']:.2f}")
# [book:calibration-end]
