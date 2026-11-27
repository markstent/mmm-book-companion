"""Chapter 29 lab: score new weeks against a predictive interval and fire the run-of-four rule.

The lab uses a stand-in for the fitted model's posterior predictive draws: a weekly
forecast of GBP 900,000 with GBP 20,000 of noise, the typical week of Chapters 2 and 29.
It needs no sampling, and the rule works the same on a fitted model's predictive draws.
"""

import numpy as np

from mmm_book import RANDOM_SEED

CENTRE = 900_000.0
NOISE_SD = 20_000.0
RUN = 4  # consecutive misses on the same side that fire the rule


def predictive_interval(n_draws: int = 4_000, level: float = 0.90, seed: int = RANDOM_SEED):
    """The central ``level`` interval of the stand-in predictive draws."""
    rng = np.random.default_rng(seed)
    draws = CENTRE + rng.normal(0.0, NOISE_SD, n_draws)
    tail = (1 - level) / 2
    return tuple(np.quantile(draws, [tail, 1 - tail]))


def first_alarm(y: np.ndarray, low: float, high: float, run: int = RUN) -> int | None:
    """The week (counting from 1) in which ``run`` misses in a row on one side complete."""
    below = above = 0
    for week, value in enumerate(y, start=1):
        below = below + 1 if value < low else 0
        above = above + 1 if value > high else 0
        if max(below, above) == run:
            return week
    return None


def new_weeks(n_weeks: int = 26, shift: float = 0.0, start: int = 13, seed: int = 1) -> np.ndarray:
    """New weeks from the same business, with ``shift`` pounds a week added from ``start``."""
    rng = np.random.default_rng(seed)
    y = CENTRE + rng.normal(0.0, NOISE_SD, n_weeks)
    y[start - 1 :] += shift
    return y


def false_alarm_rate(n_sims: int = 20_000, n_weeks: int = 52, seed: int = RANDOM_SEED) -> float:
    """Share of 52-week stretches with no change at all in which the rule still fires."""
    low, high = predictive_interval()
    rng = np.random.default_rng(seed)
    y = CENTRE + rng.normal(0.0, NOISE_SD, (n_sims, n_weeks))
    return float(np.mean([first_alarm(row, low, high) is not None for row in y]))


def main() -> dict:
    low, high = predictive_interval()
    print(f"90% predictive interval: {low:,.0f} to {high:,.0f}")
    print(f"chance of four misses in a row on a fixed side: {0.05**RUN:.1e}")
    steady = first_alarm(new_weeks(), low, high)
    print(f"no change in the business: alarm in week {steady}")
    shifted = first_alarm(new_weeks(shift=-50_000.0), low, high)
    print(f"revenue 50,000 a week lower from week 13: alarm in week {shifted}")
    rate = false_alarm_rate()
    print(f"false alarms in 52 steady weeks: {rate:.2%}")
    return {"steady": steady, "shifted": shifted, "false_alarm_rate": rate}


if __name__ == "__main__":
    main()
