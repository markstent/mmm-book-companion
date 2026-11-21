"""Recover channel ROAS from the reference model and compare to ground truth.

This recovery is loose by design: one year of two regions is thin evidence for
18 parameters, and Meta and YouTube move together, so the recovered ROAS will
not closely match GROUND_TRUTH (on the full 156-week, 18-region panel television
is recovered but Meta and YouTube are not). That gap is why ex_calibration.py
calibrates to an experiment instead of trusting the observational fit alone.
"""

# [book:recovery_imports-start]
import pymc as pm

from mmm_book import RANDOM_SEED
from mmm_book.reference import REVENUE_SCALE, build_features, national_aggregate, reference_model
from mmm_book.sedge import CHANNELS, GROUND_TRUTH, TV_COST_PER_GRP, generate_sedge

# [book:recovery_imports-end]

# [book:recovery-start]
_MEDIA_COL = {
    "tv": "tv_grps", "search": "search_spend", "meta": "meta_spend",
    "youtube": "youtube_spend", "ooh": "ooh_spend",
}

df = generate_sedge(n_weeks=52, n_regions=2)
nat = national_aggregate(df)
model = reference_model(nat)

with model:
    idata = pm.sample(
        draws=200, tune=200, chains=2, cores=1,
        target_accept=0.9, random_seed=RANDOM_SEED, progressbar=False,
    )

feat = build_features(nat)
print(f"{'channel':<10}{'recovered':>12}{'ground truth':>15}")
for ch in CHANNELS:
    beta_mean = float(idata.posterior[f"beta_{ch}"].mean())
    raw = nat[_MEDIA_COL[ch]].to_numpy(dtype=float)
    spend = raw * TV_COST_PER_GRP if ch == "tv" else raw
    recovered = beta_mean * feat[ch].sum() * REVENUE_SCALE / spend.sum()
    print(f"{ch:<10}{recovered:>12.2f}{GROUND_TRUTH[ch]:>15.2f}")
# [book:recovery-end]
