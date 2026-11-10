"""Shared national-aggregate reference MMM used across the PyMC labs.

Lab 15 builds this model inline, in stages, as the teaching origin: a
national weekly aggregate of Sedge Shoes, correctly-specified adstock and
saturation features (using the same transform hyperparameters the Sedge
generator used to build the data), and a linear-in-betas revenue model
with Chapter 15's baseline, non-media treatment and control terms. Later
labs reuse the same construction rather than re-deriving it, so this
module factors it out into small, composable pieces:

``national_aggregate`` collapses the region-week panel to one row per week.
``build_seasonality``, ``build_events``, ``build_treatments`` and
``build_controls`` build the non-media regressors, and ``build_features``
each channel's adstocked, saturated feature. ``reference_model`` assembles
the PyMC model itself (but does not sample it, so callers can extend,
inspect, or sample the model however they need).
"""

import numpy as np
import pandas as pd
import pymc as pm

from mmm_book.adstock import geometric_adstock
from mmm_book.saturation import hill_saturation
from mmm_book.sedge import ADSTOCK_ALPHA, CHANNELS, HALF_MULT, HILL_SLOPE

REVENUE_SCALE = 1e6

# Length of the seasonal cycle in weeks (the book's P).
YEAR_WEEKS = 52.18

# Prior scale of each channel's average weekly contribution, as a share of
# mean weekly revenue: Lab 17's prior predictive check chooses it.
BETA_PRIOR_SHARE = 0.1

# Columns that are averaged, not summed, when the panel is collapsed to a
# national weekly series.
AVERAGED_COLUMNS = [
    "price",
    "promo_depth",
    "distribution",
    "stock",
    "temperature",
    "rainfall",
    "launch",
    "category_search",
]

# The column each channel's feature is built from. Paid Search uses generic
# impressions: brand search spend follows demand rather than creating it
# (Chapter 35), so a feature built on total Search spend would credit Search
# with demand it only collects. Search's ROAS is still quoted on total
# Search spend.
DRIVER_COL = {
    "tv": "tv_grps",
    "search": "search_generic_impr",
    "meta": "meta_spend",
    "youtube": "youtube_spend",
    "ooh": "ooh_spend",
}


def national_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a Sedge region-week panel to a national weekly series.

    Sums media, brand search volume and revenue across regions, giving one
    row per week. The controls (price, promotion depth, distribution, stock,
    weather, the launch indicators and the category search index) are
    averaged across regions rather than summed, so they stay on their
    natural scale.
    """
    nat = df.groupby("week", as_index=False).sum(numeric_only=True).sort_values("week")
    means = df.groupby("week")[AVERAGED_COLUMNS].mean().sort_index()
    for col in means.columns:
        nat[col] = means[col].to_numpy()
    return nat


def _standardise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    # A regressor that never varies in the window (no launch in a one-year
    # slice, say) carries no information; keep it at zero rather than 0 / 0.
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def build_seasonality(nat: pd.DataFrame) -> dict[str, np.ndarray]:
    """One annual Fourier pair, the seasonal part of the baseline.

    The book writes seasonality as ``a_n sin(2 pi n t / P) + b_n cos(...)``
    with ``P`` = 52.18 weeks; the reference model keeps the first wave
    (``n`` = 1), so ``sin`` carries ``a_1`` and ``cos`` carries ``b_1``.
    """
    angle = 2 * np.pi * nat["week"].to_numpy(dtype=float) / YEAR_WEEKS
    return {"sin": np.sin(angle), "cos": np.cos(angle)}


def build_events(nat: pd.DataFrame) -> dict[str, np.ndarray]:
    """Known events in the baseline: the product-launch indicator (1 in launch weeks)."""
    return {"launch": nat["launch"].to_numpy(dtype=float)}


def build_treatments(nat: pd.DataFrame) -> dict[str, np.ndarray]:
    """Non-media treatments, each standardised: promotion depth.

    Promotion is the book's non-media treatment ``Q_j`` with effect
    ``delta_j``: measured for its own sake, entered without adstock or
    saturation, and kept apart from the controls.
    """
    return {"promo": _standardise(nat["promo_depth"])}


def build_controls(nat: pd.DataFrame) -> dict[str, np.ndarray]:
    """Controls ``Z_c`` for the reference model, each standardised.

    Log price, distribution, stock availability, temperature, rainfall and
    the log category search index: the control set Chapter 34 lists for
    Sedge. Without them the media betas absorb price swings, weather and
    category demand. Seasonality and launches belong to the baseline
    (``build_seasonality``, ``build_events``) and promotion is a non-media
    treatment (``build_treatments``), as in Chapter 15.
    """
    return {
        "log_price": _standardise(np.log(nat["price"])),
        "distribution": _standardise(nat["distribution"]),
        "stock": _standardise(nat["stock"]),
        "temperature": _standardise(nat["temperature"]),
        "rainfall": _standardise(nat["rainfall"]),
        "category": _standardise(np.log(nat["category_search"])),
    }


def saturation_means(nat: pd.DataFrame) -> dict[str, float]:
    """Mean of each channel's raw (unscaled) Hill feature on ``nat``.

    ``build_features`` divides each saturated feature by this mean, so a
    beta is the channel's average weekly contribution in revenue-scale
    units. Response-curve code divides a raw Hill value by the same mean
    to stay on the beta's scale.
    """
    return {ch: float(_raw_feature(nat, ch).mean()) for ch in CHANNELS}


def _raw_feature(nat: pd.DataFrame, ch: str) -> np.ndarray:
    raw = nat[DRIVER_COL[ch]].to_numpy(dtype=float)
    ad = geometric_adstock(raw, alpha=ADSTOCK_ALPHA[ch], normalize=True)
    return hill_saturation(ad, half=HALF_MULT[ch] * np.mean(ad), slope=HILL_SLOPE[ch])


def build_features(nat: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build each channel's adstocked, saturated feature from a national aggregate.

    Uses the same ``ADSTOCK_ALPHA``, ``HILL_SLOPE`` and ``HALF_MULT`` per
    channel that the Sedge generator used to build the data, so the
    feature is correctly specified. Each returned array is the
    Hill-saturated adstock divided by its own mean, so it averages one and
    is aligned row-for-row with ``nat``. A channel's beta is therefore its
    average weekly contribution (in ``REVENUE_SCALE`` units) rather than the
    book's weekly contribution at full saturation; the two differ by the
    channel's average Hill value, and the average keeps a single sensible
    prior scale across channels with very different curves.
    """
    feat = {}
    for ch in CHANNELS:
        raw_feature = _raw_feature(nat, ch)
        feat[ch] = raw_feature / raw_feature.mean()
    return feat


def reference_model(
    nat: pd.DataFrame, include_media: bool = True, n_train: int | None = None
) -> pm.Model:
    """Build the reference national-aggregate Bayesian MMM.

    Revenue (rescaled by ``REVENUE_SCALE``) is modelled as Chapter 15
    writes it: a baseline (intercept, a linear trend over the standardised
    week index, one annual Fourier pair and the launch event), the
    promotion treatment, the controls of ``build_controls``, and a sum of
    channel contributions, each a HalfNormal beta multiplied by that
    channel's feature.

    The priors are the ones Lab 17 chooses with prior predictive checks,
    set on revenue's own scale so they mean the same thing for any slice of
    the data: each channel's average weekly contribution is
    ``HalfNormal(BETA_PRIOR_SHARE * mean revenue)``, so a single channel is
    unlikely to explain more than about a fifth of revenue, and every other
    coefficient, and the noise, has a prior scale equal to the standard
    deviation of weekly revenue.

    This is the labs' teaching model: national rather than regional, no
    hierarchy, and the adstock and Hill hyperparameters fixed at their true
    values rather than estimated. The case-study model of Chapters 35 to
    39, which is regional, estimates the transforms and puts priors on
    ROAS, is ``mmm_book.case_model``.

    ``include_media=False`` builds the first stage of Chapter 15's staged
    build: baseline, treatment and controls only. With ``n_train`` set, the
    likelihood (and the prior scales) use only the first ``n_train`` weeks,
    and the model also records ``mu``, the expected revenue for every week,
    so the held-out weeks can be predicted from the posterior. Features,
    controls and the week index are built on the full ``nat``: they are
    known in advance, like a media plan, and carry no revenue information.
    Returns the built model without sampling it.
    """
    n_weeks = len(nat)
    n_fit = n_weeks if n_train is None else int(n_train)
    if not 0 < n_fit <= n_weeks:
        raise ValueError("n_train must be between 1 and the number of weeks")

    feat = build_features(nat)
    season = build_seasonality(nat)
    events = build_events(nat)
    treatments = build_treatments(nat)
    controls = build_controls(nat)

    y = nat["revenue"].to_numpy(dtype=float) / REVENUE_SCALE
    y_fit = y[:n_fit]
    scale = float(y_fit.std())
    week = nat["week"].to_numpy(dtype=float)
    week_c = (week - week.mean()) / week.std()

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=y_fit.mean(), sigma=scale)
        trend = pm.Normal("trend", 0, scale)
        season_a = pm.Normal("season_a", 0, scale)
        season_b = pm.Normal("season_b", 0, scale)
        omega = {name: pm.Normal(f"omega_{name}", 0, scale) for name in events}
        delta = {name: pm.Normal(f"delta_{name}", 0, scale) for name in treatments}
        gamma = {name: pm.Normal(f"gamma_{name}", 0, scale) for name in controls}
        mu = (
            intercept
            + trend * week_c
            + season_a * season["sin"]
            + season_b * season["cos"]
            + sum(omega[name] * events[name] for name in events)
            + sum(delta[name] * treatments[name] for name in treatments)
            + sum(gamma[name] * controls[name] for name in controls)
        )
        if include_media:
            beta_sigma = BETA_PRIOR_SHARE * float(y_fit.mean())
            betas = {ch: pm.HalfNormal(f"beta_{ch}", sigma=beta_sigma) for ch in CHANNELS}
            mu = mu + sum(betas[ch] * feat[ch] for ch in CHANNELS)
        sigma = pm.HalfNormal("sigma", sigma=scale)
        if n_train is not None:
            mu = pm.Deterministic("mu", mu)
        pm.Normal("y", mu=mu[:n_fit], sigma=sigma, observed=y_fit)

    return model
