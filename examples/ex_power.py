"""Chapter 2 lab: how often a four-week campaign is detected against weekly noise."""

from math import erf, sqrt

import numpy as np

from mmm_book import RANDOM_SEED

Z_CRITICAL = 1.96  # the one-sided threshold of Primer 2's power arithmetic


def simulate_power(
    uplift: float = 20_000.0,
    noise_sd: float = 20_000.0,
    weeks: int = 4,
    n_sims: int = 200_000,
    seed: int = RANDOM_SEED,
) -> float:
    """Share of simulated campaigns whose average effect clears 1.96 standard errors."""
    rng = np.random.default_rng(seed)
    weekly = uplift + rng.normal(0.0, noise_sd, (n_sims, weeks))
    standard_error = noise_sd / sqrt(weeks)
    return float((weekly.mean(axis=1) / standard_error > Z_CRITICAL).mean())


def exact_power(uplift: float, noise_sd: float, weeks: int) -> float:
    """The same probability from the normal distribution, for comparison."""
    z = uplift / (noise_sd / sqrt(weeks)) - Z_CRITICAL
    return 0.5 * (1 + erf(z / sqrt(2)))


def main() -> float:
    print(f"{'uplift':>8}{'noise':>8}{'weeks':>7}{'simulated':>11}{'exact':>8}")
    for uplift, noise, weeks in [
        (20_000, 20_000, 4),  # the chapter's campaign
        (40_000, 20_000, 4),
        (20_000, 10_000, 4),
        (20_000, 20_000, 8),
        (20_000, 20_000, 16),
    ]:
        power = simulate_power(uplift, noise, weeks)
        exact = exact_power(uplift, noise, weeks)
        print(f"{uplift:>8,}{noise:>8,}{weeks:>7}{power:>11.1%}{exact:>8.1%}")
    return simulate_power()


if __name__ == "__main__":
    print(f"power of the chapter's campaign: {main():.1%}")
