"""Calibrate Meta to the Chapter 36 geo holdout (GBP 138k lift from GBP 77k of withheld spend)."""

import pymc as pm

from mmm_book import RANDOM_SEED
from mmm_book.experiment import (
    EXPERIMENT_LIFT_GBP,
    EXPERIMENT_ROAS,
    EXPERIMENT_SE_GBP,
    EXPERIMENT_WITHHELD_SPEND_GBP,
)
from mmm_book.reference import REVENUE_SCALE, build_features, national_aggregate, reference_model
from mmm_book.sedge import GROUND_TRUTH, generate_sedge

nat = national_aggregate(generate_sedge())
feat_total = float(build_features(nat)["meta"].sum())
spend_total = float(nat["meta_spend"].to_numpy(dtype=float).sum())

model = reference_model(nat)
with model:
    implied_roas = pm.Deterministic(
        "implied_meta_roas", model["beta_meta"] * feat_total * REVENUE_SCALE / spend_total
    )
    # Observed lift 138,000 (se 45,000) against the lift implied by the 77,000 of spend
    # that was switched off. A holdout reads an average-type return on that slice.
    pm.Normal(
        "experiment",
        mu=implied_roas * EXPERIMENT_WITHHELD_SPEND_GBP,
        sigma=EXPERIMENT_SE_GBP,
        observed=EXPERIMENT_LIFT_GBP,
    )
    idata = pm.sample(
        draws=200,
        tune=200,
        chains=2,
        cores=1,
        target_accept=0.9,
        random_seed=RANDOM_SEED,
        progressbar=False,
    )

calibrated = float(idata.posterior["implied_meta_roas"].mean())
print(f"holdout result:       {EXPERIMENT_ROAS:.2f} ROAS")
print(f"calibrated ROAS mean: {calibrated:.2f}")
print(f"ledger ROAS (truth):  {GROUND_TRUTH['meta']:.2f}")
