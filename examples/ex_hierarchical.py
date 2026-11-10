"""Minimal partial pooling: region betas are a channel effect times exp(tau * z) (Lab 19)."""

import numpy as np
import pymc as pm

from mmm_book import RANDOM_SEED
from mmm_book.reference import REVENUE_SCALE, build_features
from mmm_book.sedge import CHANNELS, generate_sedge

df = generate_sedge(n_weeks=52, n_regions=2)
regions = sorted(df["region"].unique())
frames = {r: df[df["region"] == r].sort_values("week") for r in regions}
feat = np.stack([np.stack([build_features(frames[r])[ch] for ch in CHANNELS]) for r in regions])
y = np.stack([frames[r]["revenue"].to_numpy(dtype=float) / REVENUE_SCALE for r in regions])

with pm.Model(coords={"region": regions, "channel": CHANNELS}) as model:
    level = pm.Normal("level", mu=y.mean(axis=1), sigma=y.std(), dims="region")
    mu_ch = pm.HalfNormal("mu_ch", sigma=0.1 * y.mean(), dims="channel")
    tau = pm.HalfNormal("tau", sigma=0.15, dims="channel")
    z = pm.Normal("z", 0.0, 1.0, dims=("region", "channel"))
    beta = pm.Deterministic("beta", mu_ch * pm.math.exp(tau * z), dims=("region", "channel"))
    sigma = pm.HalfNormal("sigma", sigma=y.std())
    mu = level[:, None] + (beta[:, :, None] * feat).sum(axis=1)
    pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    idata = pm.sample(
        draws=200, tune=200, chains=2, cores=1,
        target_accept=0.9, random_seed=RANDOM_SEED, progressbar=False,
    )

print(idata.posterior["mu_ch"].mean(dim=("chain", "draw")).to_pandas())
