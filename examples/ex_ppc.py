"""Posterior predictive check: sample, then compare to observed revenue."""

import numpy as np
import pymc as pm

from mmm_book import RANDOM_SEED
from mmm_book.reference import national_aggregate, reference_model
from mmm_book.sedge import generate_sedge

df = generate_sedge(n_weeks=52, n_regions=2)
nat = national_aggregate(df)
model = reference_model(nat)

with model:
    idata = pm.sample(
        draws=200, tune=200, chains=2, cores=1,
        target_accept=0.9, random_seed=RANDOM_SEED, progressbar=False,
    )
    pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=RANDOM_SEED)

y_obs = idata.observed_data["y"].to_numpy()
pp_mean = idata.posterior_predictive["y"].mean(dim=("chain", "draw")).to_numpy()
rmse = float(np.sqrt(np.mean((y_obs - pp_mean) ** 2)))
print(f"observed mean revenue: {y_obs.mean():.3f}")
print(f"posterior predictive mean: {pp_mean.mean():.3f}")
print(f"posterior predictive RMSE: {rmse:.3f}")
