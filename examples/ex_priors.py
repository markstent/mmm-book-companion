"""Prior predictive sample on the reference model."""

# [book:priors_imports-start]
import pymc as pm

from mmm_book import RANDOM_SEED
from mmm_book.reference import (
    national_aggregate,
    reference_model,
)
from mmm_book.sedge import generate_sedge

# [book:priors_imports-end]

# [book:priors-start]
df = generate_sedge(n_weeks=52, n_regions=2)
nat = national_aggregate(df)
model = reference_model(nat)

with model:
    prior = pm.sample_prior_predictive(
        draws=200,
        random_seed=RANDOM_SEED,
    )

y_prior = prior.prior_predictive["y"].to_numpy()
print("prior predictive y shape:", y_prior.shape)
mean, sd = y_prior.mean(), y_prior.std()
print(f"prior predictive mean: {mean:.2f}, sd: {sd:.2f}")
# [book:priors-end]
