"""Response curves and a small budget allocator.

Two kinds of curve share one interface (``current_money``,
``total_budget``, ``money_contribution``, ``average_return``,
``marginal_return``), so ``greedy_allocate`` runs on either.

``PosteriorCurves`` is the one Part IX uses. It is built by
``case_model.posterior_curves`` from a fitted case-study model: every
posterior draw has its own adstock retention, half-saturation point and
regional scales, so every draw has its own curve, and a plan can be scored
on each of them. Its ``money_contribution`` is the average over draws, the
posterior mean curve.

``ResponseCurves`` is the teaching version that Chapters 27 and 28 and the
labs use. It turns a national weekly aggregate and one beta per channel
into expected weekly contribution, with the generator's own adstock and
saturation shapes, the same features as ``mmm_book.reference``.
``ledger_betas`` gives the betas that make those curves reproduce the
built-in true average returns exactly (2.4x Television, 2.7x Search, 1.75x
Meta, 2.0x YouTube, 1.5x OOH), so Chapters 27 and 28 can do their
arithmetic on one known set of curves.

``greedy_allocate`` is the dependency-light greedy marginal allocator of
lab 27, extended with per-channel lower and upper bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmm_book.adstock import geometric_adstock
from mmm_book.reference import DRIVER_COL, REVENUE_SCALE, build_features, saturation_means
from mmm_book.saturation import hill_saturation
from mmm_book.sedge import (
    ADSTOCK_ALPHA,
    CHANNELS,
    GROUND_TRUTH,
    HALF_MULT,
    HILL_SLOPE,
    TV_COST_PER_GRP,
)

MEDIA_COL = {
    "tv": "tv_grps",
    "search": "search_spend",
    "meta": "meta_spend",
    "youtube": "youtube_spend",
    "ooh": "ooh_spend",
}


def to_money(ch: str, native: float) -> float:
    """GRPs to GBP-equivalent for Television, identity for everything else."""
    return native * TV_COST_PER_GRP if ch == "tv" else native


def to_native(ch: str, money: float) -> float:
    return money / TV_COST_PER_GRP if ch == "tv" else money


def betas_for_roas(nat: pd.DataFrame, roas: dict[str, float]) -> dict[str, float]:
    """Betas that give each channel the average ROAS in ``roas`` on ``nat``.

    The curves keep the generator's adstock and saturation shapes; only
    their height changes. Passing a fitted model's posterior mean ROAS gives
    curves at the fitted level of return.
    """
    feat = build_features(nat)
    betas = {}
    for ch in CHANNELS:
        spend_total = to_money(ch, float(nat[MEDIA_COL[ch]].sum()))
        betas[ch] = roas[ch] * spend_total / (float(feat[ch].sum()) * REVENUE_SCALE)
    return betas


def ledger_betas(nat: pd.DataFrame) -> dict[str, float]:
    """Betas that give each channel exactly the book's average ROAS on ``nat``."""
    return betas_for_roas(nat, GROUND_TRUTH)


class ResponseCurves:
    """Average weekly contribution (GBP) as a function of a channel's spend level.

    A budget change scales the channel's whole historical weekly plan, so its
    flighting is kept: Television stays concentrated around the spring races
    and Christmas at any budget. ``native_level`` is the plan's average
    weekly level (GRPs for Television, GBP for everything else). At the
    current level the curve's average return is the channel's realised
    average ROAS, and its slope is the marginal ROAS of a proportional rise
    in every week's spend, the quantity ``sedge.true_marginal_roas`` reports.
    """

    def __init__(self, nat: pd.DataFrame, beta: dict[str, float]):
        self.beta = dict(beta)
        self.hill_mean = saturation_means(nat)
        self.half = {}
        self.current_native = {}
        self._driver = {}
        for ch in CHANNELS:
            driver = nat[DRIVER_COL[ch]].to_numpy(dtype=float)
            adstocked = geometric_adstock(driver, alpha=ADSTOCK_ALPHA[ch], normalize=True)
            self.half[ch] = HALF_MULT[ch] * float(np.mean(adstocked))
            self.current_native[ch] = float(nat[MEDIA_COL[ch]].mean())
            self._driver[ch] = driver
        self.current_money = {ch: to_money(ch, self.current_native[ch]) for ch in CHANNELS}
        self.total_budget = sum(self.current_money.values())

    def contribution(self, ch: str, native_level: float) -> float:
        # Spend scales the channel's driver (generic impressions for Search)
        # in proportion, week by week.
        series = self._driver[ch] * max(native_level, 0.0) / self.current_native[ch]
        adstocked = geometric_adstock(series, alpha=ADSTOCK_ALPHA[ch], l_max=12, normalize=True)
        saturated = hill_saturation(adstocked, half=self.half[ch], slope=HILL_SLOPE[ch]).mean()
        return self.beta[ch] * saturated / self.hill_mean[ch] * REVENUE_SCALE

    def money_contribution(self, ch: str, money: float) -> float:
        return self.contribution(ch, to_native(ch, money))

    def average_return(self, ch: str) -> float:
        return self.money_contribution(ch, self.current_money[ch]) / self.current_money[ch]

    def marginal_return(self, ch: str, rel_step: float = 0.01) -> float:
        eps = self.current_money[ch] * rel_step
        hi = self.money_contribution(ch, self.current_money[ch] + eps)
        lo = self.money_contribution(ch, self.current_money[ch] - eps)
        return (hi - lo) / (2 * eps)

    def total_contribution(self, money: dict[str, float]) -> float:
        return sum(self.money_contribution(ch, money[ch]) for ch in CHANNELS)


class PosteriorCurves:
    """Response curves for every posterior draw of a fitted model.

    ``adstocked[ch]`` is a (draws, regions, weeks) array of the adstocked
    treatment under the historical plan, ``half[ch]`` the (draws, regions)
    half-saturation points in the same units, and ``beta[ch]`` the
    (draws, regions) scales in pounds of weekly revenue at full saturation.
    ``current[ch]`` is the channel's average weekly national spend (pounds;
    Television's GRPs are priced at ``TV_COST_PER_GRP``).

    Spending ``money`` a week scales the whole historical plan of the
    channel by ``money / current`` in every region and week, which scales
    the adstocked treatment by the same factor (adstock is linear). Each
    draw's curve is tabulated once on a grid of that factor, from 0 to
    ``max_scale`` times current spend, and interpolated with a cubic spline.
    """

    def __init__(self, adstocked, half, beta, current, max_scale: float = 4.0, n_grid: int = 321):
        grid = np.linspace(0.0, max_scale, n_grid)
        tables = {}
        for ch in CHANNELS:
            a, h, b = adstocked[ch], half[ch], beta[ch]
            n_weeks = a.shape[-1]
            table = np.empty((a.shape[0], n_grid))
            for g, scale in enumerate(grid):
                sa = scale * a
                table[:, g] = np.einsum("dr,drt->d", b, sa / (sa + h[:, :, None])) / n_weeks
            tables[ch] = table
        self._set(grid, tables, current)

    def _set(self, grid, tables, current) -> None:
        from scipy.interpolate import CubicSpline

        self.grid = grid
        self.max_scale = float(grid[-1])
        self.tables = tables
        self.current_money = dict(current)
        self.total_budget = sum(self.current_money.values())
        self._spline = {ch: CubicSpline(grid, t, axis=1) for ch, t in tables.items()}
        self._mean_spline = {ch: CubicSpline(grid, t.mean(axis=0)) for ch, t in tables.items()}
        self.n_draws = next(iter(tables.values())).shape[0]

    def subset(self, keep: np.ndarray) -> PosteriorCurves:
        """The same curves restricted to the draws where ``keep`` is true."""
        out = PosteriorCurves.__new__(PosteriorCurves)
        out._set(self.grid, {ch: t[keep] for ch, t in self.tables.items()}, self.current_money)
        return out

    def _scale(self, ch: str, money: float) -> float:
        scale = max(money, 0.0) / self.current_money[ch]
        if scale > self.max_scale + 1e-9:
            raise ValueError(f"{ch}: {scale:.2f} times current spend is beyond the curve grid")
        return scale

    def draws(self, ch: str, money: float) -> np.ndarray:
        """Weekly contribution (pounds) at ``money`` a week, one value per draw."""
        return self._spline[ch](self._scale(ch, money))

    def money_contribution(self, ch: str, money: float) -> float:
        """Posterior mean weekly contribution at ``money`` a week."""
        return float(self._mean_spline[ch](self._scale(ch, money)))

    def average_return(self, ch: str) -> float:
        return self.money_contribution(ch, self.current_money[ch]) / self.current_money[ch]

    def marginal_draws(self, ch: str, rel_step: float = 0.01) -> np.ndarray:
        """Return on a 1 per cent rise in every week's spend, one value per draw."""
        m = self.current_money[ch]
        return (self.draws(ch, m * (1 + rel_step)) - self.draws(ch, m)) / (m * rel_step)

    def marginal_return(self, ch: str, rel_step: float = 0.01) -> float:
        return float(self.marginal_draws(ch, rel_step).mean())

    def total_contribution(self, money: dict[str, float]) -> float:
        return sum(self.money_contribution(ch, money[ch]) for ch in CHANNELS)

    def gain_draws(self, plan: dict[str, float], base: dict | None = None) -> np.ndarray:
        """Change in weekly contribution from ``base`` (default: current) to ``plan``, per draw."""
        base = self.current_money if base is None else base
        return sum(self.draws(ch, plan[ch]) - self.draws(ch, base[ch]) for ch in CHANNELS)


def greedy_allocate(
    curves: ResponseCurves,
    lower: dict[str, float] | None = None,
    upper: dict[str, float] | None = None,
    total: float | None = None,
    n_steps: int = 4000,
) -> dict[str, float]:
    """Spread ``total`` (default: the current budget) in small equal steps.

    Each step goes to the channel with the highest marginal return that is
    still below its upper bound. Allocation starts at the lower bounds, so
    bounds are honoured exactly (to within one step).
    """
    total = curves.total_budget if total is None else total
    lower = lower or {ch: 0.0 for ch in CHANNELS}
    upper = upper or {ch: np.inf for ch in CHANNELS}
    alloc = {ch: float(lower[ch]) for ch in CHANNELS}
    remaining = total - sum(alloc.values())
    if remaining < 0:
        raise ValueError("lower bounds exceed the total budget")
    step = remaining / n_steps
    for _ in range(n_steps):
        gains = {
            ch: (
                curves.money_contribution(ch, alloc[ch] + step)
                - curves.money_contribution(ch, alloc[ch])
            )
            / step
            for ch in CHANNELS
            if alloc[ch] + step <= upper[ch] + 1e-9
        }
        if not gains:
            break
        alloc[max(gains, key=gains.get)] += step
    return alloc


def band(curves: ResponseCurves, pct: float) -> tuple[dict[str, float], dict[str, float]]:
    """Lower and upper bounds of plus or minus ``pct`` around current spend."""
    lo = {ch: curves.current_money[ch] * (1 - pct) for ch in CHANNELS}
    hi = {ch: curves.current_money[ch] * (1 + pct) for ch in CHANNELS}
    return lo, hi


# The risk-aware operational plan of Chapter 38 (annual GBP): Search
# 1.30m, Meta 1.25m, YouTube 0.85m, Television 1.65m, OOH 0.95m.
BOOK_PLAN_ANNUAL = {
    "search": 1_300_000.0,
    "meta": 1_250_000.0,
    "youtube": 850_000.0,
    "tv": 1_650_000.0,
    "ooh": 950_000.0,
}

# The same plan's constraints as bounds on the annual GBP budget. Search
# keeps a strategic floor (-100k), Television has an inventory ceiling
# (+250k), the two channels no experiment has tested (YouTube, OOH) may
# grow by at most 50k each, and Meta may fall by at most 20 per cent (to 1.20m), the
# first-year rule for mature channels in Chapter 27.
BOOK_PLAN_ANNUAL_BOUNDS = {
    "search": (1_300_000.0, 1_400_000.0),
    "tv": (1_400_000.0, 1_650_000.0),
    "youtube": (800_000.0, 850_000.0),
    "ooh": (900_000.0, 950_000.0),
    "meta": (1_200_000.0, 1_500_000.0),
}


def book_plan_bounds(curves: ResponseCurves) -> tuple[dict[str, float], dict[str, float]]:
    """Weekly bounds for the Chapter 38 operational plan.

    The annual bounds are converted to weekly figures and scaled by the
    ratio of the simulated budget to the book's GBP 6.0m, so the bounds sit
    on the same total as ``curves``.
    """
    scale = curves.total_budget * 52.0 / 6_000_000.0
    lo = {ch: BOOK_PLAN_ANNUAL_BOUNDS[ch][0] / 52.0 * scale for ch in CHANNELS}
    hi = {ch: BOOK_PLAN_ANNUAL_BOUNDS[ch][1] / 52.0 * scale for ch in CHANNELS}
    return lo, hi


def book_plan_weekly(curves: ResponseCurves) -> dict[str, float]:
    """The Chapter 38 plan as weekly GBP-equivalent, on the same total as ``curves``."""
    scale = curves.total_budget * 52.0 / 6_000_000.0
    return {ch: BOOK_PLAN_ANNUAL[ch] / 52.0 * scale for ch in CHANNELS}
