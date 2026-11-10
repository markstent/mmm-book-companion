"""Fit the reference model and summarise the channel betas."""

import arviz as az
import pymc as pm

from mmm_book import RANDOM_SEED
from mmm_book.reference import national_aggregate, reference_model
from mmm_book.sedge import CHANNELS, generate_sedge

df = generate_sedge(n_weeks=52, n_regions=2)
nat = national_aggregate(df)
model = reference_model(nat)

with model:
    idata = pm.sample(
        draws=200,
        tune=200,
        chains=2,
        cores=1,
        target_accept=0.9,
        random_seed=RANDOM_SEED,
        progressbar=False,
    )

beta_names = [f"beta_{ch}" for ch in CHANNELS]
print(az.summary(idata, var_names=beta_names, ci_prob=0.9))
