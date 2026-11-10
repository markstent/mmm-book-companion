"""The Sedge Shoes case-study model of Chapters 35 to 39.

This is the model the book describes, fitted to the regional panel rather
than to a national aggregate:

* a level for each of the 18 regions, estimated separately as a log-scale
  departure from the region's own average sales (no pooling: every region
  has 156 weeks of sales to pin its own level);
* a log-linear baseline with a trend, two annual Fourier pairs and one
  uplift per product launch, each launch with its own independent
  Normal(0.05, 0.1) prior, and the agreed control set (price, promotion
  depth, distribution, stock, temperature, rainfall and the category
  search index) on the same log scale;
* geometric adstock and Hill saturation for each channel, with the adstock
  retention and the half-saturation point *estimated*, and regional media
  effects partially pooled around the national effect;
* priors on each channel's return on ad spend (ROAS) directly, so the
  priors read in business language: Television centred near 2.2x, Meta
  near 1.8x, a weak positive prior on YouTube, a Search prior that allows a
  strong return but excludes platform-attribution values, and a broad
  positive prior on OOH.

Two treatment sets exist. ``"model1"`` is Chapter 35's first model, with
total Paid Search spend as the Search treatment. ``"model2"`` is the
revised model: generic Search impressions replace total Search spend and
brand search leaves the media set. Paid Search's ROAS is quoted on total
Search spend in both, as the book quotes it.

``build_case_model`` returns an unsampled PyMC model; ``fit_case_model``
samples it (and caches the result on disk, keyed by everything that
affects the fit, so later notebooks can reuse an earlier one's run).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from mmm_book import RANDOM_SEED
from mmm_book.sedge import CHANNELS

Y_SCALE = 1e4  # revenue is modelled in tens of thousands of pounds
L_MAX = 12
FOURIER_ORDER = 2

SPEND_COL = {
    "tv": "tv_spend",
    "search": "search_spend",
    "meta": "meta_spend",
    "youtube": "youtube_spend",
    "ooh": "ooh_spend",
}

TREATMENT_COL = {
    "model1": {
        "tv": "tv_grps",
        "search": "search_spend",
        "meta": "meta_spend",
        "youtube": "youtube_spend",
        "ooh": "ooh_spend",
    },
    "model2": {
        "tv": "tv_grps",
        "search": "search_generic_impr",
        "meta": "meta_spend",
        "youtube": "youtube_spend",
        "ooh": "ooh_spend",
    },
}

# Controls enter on the log scale beside the baseline; the multiplicative ones as logs.
CONTROLS = {
    "price": lambda d: np.log(d["price"]),
    "promo_depth": lambda d: d["promo_depth"],
    "distribution": lambda d: np.log(d["distribution"]),
    "stock": lambda d: np.log(d["stock"]),
    "temperature": lambda d: d["temperature"],
    "rainfall": lambda d: d["rainfall"],
    "category_search": lambda d: np.log(d["category_search"]),
}


@dataclass(frozen=True)
class CasePriors:
    """Priors in business units.

    ``roas`` maps a channel to the (median, log-scale sd) of a LogNormal
    prior on its average ROAS. ``adstock`` maps a channel to the (a, b) of a
    Beta prior on its weekly adstock retention. ``saturation`` maps a
    channel to the (median, log-scale sd) of a LogNormal prior on its
    half-saturation point, as a multiple of the region's average weekly
    activity.
    """

    roas: dict = field(default_factory=dict)
    adstock: dict = field(default_factory=dict)
    saturation: dict = field(default_factory=dict)

    def as_json(self) -> str:
        return json.dumps(
            {"roas": self.roas, "adstock": self.adstock, "saturation": self.saturation},
            sort_keys=True,
        )


# Chapter 35, "Prior design", before the prior predictive check.
FIRST_PRIORS = CasePriors(
    roas={
        "tv": (2.2, 0.35),  # two geo tests at 2.0x and 2.4x; widened
        "search": (2.5, 0.5),  # strong returns allowed, platform values excluded
        "meta": (1.8, 0.45),  # prospecting test read ~2.3x; set below it, kept broad
        "youtube": (1.5, 0.8),  # no experiment: weak, positive
        "ooh": (1.2, 1.5),  # no causal evidence: broad, positive
    },
    adstock={
        "tv": (3.0, 3.0),
        "search": (2.0, 2.0),  # allows half-lives beyond eight weeks
        "meta": (2.0, 3.0),
        "youtube": (2.0, 3.0),
        "ooh": (3.0, 3.0),
    },
    # Search and Meta are bought in auctions, where the cheapest clicks and
    # impressions go first, so their curves are expected to bend near
    # current spend. Broadcast-style channels are expected to bend later.
    saturation={
        "tv": (3.0, 1.2),
        "search": (1.0, 0.8),
        "meta": (1.5, 1.0),
        "youtube": (3.0, 1.2),
        "ooh": (3.0, 1.2),
    },
)

# After the prior predictive check: OOH tightened so fantastical annual
# contributions get little probability, and Search's carryover moved
# towards faster decay. Television's adstock prior stays broad.
PRIORS = CasePriors(
    roas={**FIRST_PRIORS.roas, "ooh": (1.2, 0.6)},
    adstock={**FIRST_PRIORS.adstock, "search": (2.0, 6.0)},
    saturation=FIRST_PRIORS.saturation,
)


@dataclass
class CaseData:
    """Arrays the model needs, in (region, week) layout."""

    regions: list[str]
    weeks: np.ndarray
    y: np.ndarray
    drivers: dict[str, np.ndarray]
    spend: dict[str, np.ndarray]
    controls: np.ndarray
    control_names: list[str]
    events: np.ndarray
    fourier: np.ndarray
    trend: np.ndarray
    variant: str
    train_weeks: int | None = None
    control_raw: np.ndarray | None = None
    control_centre: np.ndarray | None = None
    control_scale: np.ndarray | None = None

    @property
    def n_fit(self) -> int:
        """Weeks the model learns from: the training weeks, or the whole panel."""
        return len(self.weeks) if self.train_weeks is None else self.train_weeks

    @property
    def n_years(self) -> float:
        return len(self.weeks) / 52.0

    def annual_spend(self, ch: str) -> float:
        """Average annual spend over the weeks the model learns from."""
        return float(self.spend[ch][:, : self.n_fit].sum()) / (self.n_fit / 52.0)

    def reference_activity(self, ch: str) -> np.ndarray:
        """Each region's average weekly treatment over the weeks the model learns from.

        The half-saturation point is a multiple of this level. With
        ``train_weeks`` set it uses the training weeks only, so nothing about
        the held-out weeks reaches the fit.
        """
        return self.drivers[ch][:, : self.n_fit].mean(axis=1)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for arr in (self.y, self.controls, *self.drivers.values(), *self.spend.values()):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(self.variant.encode())
        h.update(str(self.train_weeks).encode())
        return h.hexdigest()[:16]


def _panel(df: pd.DataFrame, col: str | pd.Series) -> np.ndarray:
    values = df[col] if isinstance(col, str) else col
    frame = pd.DataFrame({"region": df["region"], "week": df["week"], "v": values})
    wide = frame.pivot(index="region", columns="week", values="v")
    order = sorted(wide.index, key=lambda r: int(str(r).split("_")[-1]))
    return wide.loc[order].to_numpy(dtype=float)


def prepare_case_data(
    df: pd.DataFrame, variant: str = "model2", train_weeks: int | None = None
) -> CaseData:
    """Reshape a Sedge panel into the arrays ``build_case_model`` uses.

    Controls and the trend index are standardised with the mean and standard
    deviation of the weeks the model learns from. With ``train_weeks`` set
    those are the first ``train_weeks`` weeks only, so the scaling carries no
    information from the held-out weeks into training (Chapter 21).
    """
    if variant not in TREATMENT_COL:
        raise ValueError(f"variant must be one of {sorted(TREATMENT_COL)}")
    regions = sorted(df["region"].unique(), key=lambda r: int(str(r).split("_")[-1]))
    weeks = np.sort(df["week"].unique())
    n_fit = len(weeks) if train_weeks is None else int(train_weeks)
    if not 0 < n_fit <= len(weeks):
        raise ValueError("train_weeks must be between 1 and the number of weeks")
    raw = np.stack([_panel(df, fn(df)) for fn in CONTROLS.values()], axis=-1)
    centre = raw[:, :n_fit].mean(axis=(0, 1))
    scale = raw[:, :n_fit].std(axis=(0, 1))
    controls = (raw - centre) / scale
    # One indicator per launch: each run of consecutive launch weeks.
    launch = df.groupby("week")["launch"].max().reindex(weeks).to_numpy()
    starts = np.flatnonzero(np.diff(np.concatenate([[0.0], launch])) > 0)
    events = np.zeros((len(weeks), len(starts)))
    for e, start in enumerate(starts):
        stop = start
        while stop < len(weeks) and launch[stop] > 0:
            stop += 1
        events[start:stop, e] = 1.0
    woy = weeks % 52
    fourier = np.column_stack(
        [f(2 * np.pi * k * woy / 52) for k in range(1, FOURIER_ORDER + 1) for f in (np.sin, np.cos)]
    )
    return CaseData(
        regions=list(regions),
        weeks=weeks,
        y=_panel(df, "revenue") / Y_SCALE,
        drivers={ch: _panel(df, TREATMENT_COL[variant][ch]) for ch in CHANNELS},
        spend={ch: _panel(df, SPEND_COL[ch]) for ch in CHANNELS},
        controls=controls,
        control_names=list(CONTROLS),
        events=events,
        fourier=fourier,
        trend=(weeks - weeks[:n_fit].mean()) / weeks[:n_fit].std(),
        variant=variant,
        train_weeks=train_weeks,
        control_raw=raw,
        control_centre=centre,
        control_scale=scale,
    )


def _lag_stack(x: np.ndarray) -> np.ndarray:
    """(L_MAX, regions, weeks) array of x shifted by 0..L_MAX-1 weeks, zeros before week 0."""
    out = np.zeros((L_MAX, *x.shape))
    for lag in range(L_MAX):
        out[lag, :, lag:] = x[:, : x.shape[1] - lag]
    return out


def _adstock_hill(lags, alpha, k, ref):
    """Normalised geometric adstock then Hill (shape 1) on a lag stack."""
    w = alpha ** pt.arange(L_MAX)
    w = w / (w.sum() + 1e-12)
    adstocked = pt.tensordot(w, lags, axes=1)
    half = k * ref[:, None]
    # The tiny constant keeps 0 / 0 out of the graph when a paused week
    # meets an extreme half-saturation draw during warm-up.
    return adstocked / (adstocked + half + 1e-12)


def holdout_window(df: pd.DataFrame | None = None) -> dict:
    """Treatment regions and weeks of the Chapter 36 Meta holdout."""
    from mmm_book.experiment import (
        EXPERIMENT_COOLDOWN_WEEKS,
        EXPERIMENT_START_WEEK,
        EXPERIMENT_WEEKS,
        ledger_holdout_design,
    )

    design = ledger_holdout_design()
    return {
        "treatment_regions": design["treatment_regions"],
        "start": EXPERIMENT_START_WEEK,
        "paused_weeks": EXPERIMENT_WEEKS,
        "measured_weeks": EXPERIMENT_WEEKS + EXPERIMENT_COOLDOWN_WEEKS,
    }


def build_case_model(
    data: CaseData,
    priors: CasePriors = PRIORS,
    calibrate: bool = False,
    store_mu: bool = True,
) -> pm.Model:
    """Build (but do not sample) the Sedge case-study model.

    With ``calibrate=True`` the Chapter 36 Meta holdout enters as one more
    observation: the model pushes the pause (Meta switched off in the
    treatment regions for six weeks) through its own adstock and saturation
    curve, sums the revenue it implies was lost over the six weeks plus the
    two-week cooldown, and treats the measured £138,000 (standard error
    £45,000) as a noisy reading of that quantity.

    When ``data`` was prepared with ``train_weeks``, only those weeks enter
    the likelihood, and everything computed from the data (the scaling of
    controls and trend, each region's reference activity for the
    half-saturation point, and the spend that defines ROAS) uses them alone,
    so the remaining weeks are a true test period (Chapter 36's validation
    trains on 130 weeks and predicts the last 26).
    """
    R, T = data.y.shape
    coords = {
        "region": data.regions,
        "week": data.weeks,
        "channel": CHANNELS,
        "control": data.control_names,
        "launch": [f"launch_{e + 1}" for e in range(data.events.shape[1])],
        "fourier": [f"{f}{k}" for k in range(1, FOURIER_ORDER + 1) for f in ("sin", "cos")],
    }
    n = data.n_fit
    log_y_region = np.log(data.y[:, :n].mean(axis=1))

    with pm.Model(coords=coords) as model:
        # baseline: one independent level per region
        # Each region's level, as a log-scale departure from its own average
        # sales. Every region has years of weekly data, so its level is pinned
        # by its own sales and pooling the levels would buy nothing; written
        # this way the levels sample without the slow common drift that a
        # shared mean and spread create (Chapter 18).
        level_dev = pm.Normal("level_dev", 0.0, 0.5, dims="region")
        level = pm.Deterministic("level", pt.exp(log_y_region + level_dev), dims="region")
        trend = pm.Normal("trend", 0.0, 0.2)
        season = pm.Normal("season", 0.0, 0.2, dims="fourier")
        gamma = pm.Normal("gamma", 0.0, 0.2, dims="control")
        # Launches: each its own uplift, with a wide prior centred on a 5 per
        # cent lift. Only three launches exist, too few to learn a shared
        # spread, so a launch the data have not seen keeps a wide range
        # around that centre (Chapter 36).
        launch_effect = pm.Normal("launch_effect", 0.05, 0.1, dims="launch")
        log_mult = (
            trend * data.trend[None, :]
            + pt.dot(data.fourier, season)[None, :]
            + pt.dot(data.events, launch_effect)[None, :]
            + pt.tensordot(data.controls, gamma, axes=1)
        )
        baseline = level[:, None] * pt.exp(log_mult)

        # media: priors on ROAS, adstock and saturation estimated
        contribution = []
        for ch in CHANNELS:
            median, sd = priors.roas[ch]
            a, b = priors.adstock[ch]
            k_median, k_sd = priors.saturation[ch]
            roas = pm.LogNormal(f"roas_{ch}", np.log(median), sd)
            alpha = pm.Beta(f"alpha_{ch}", a, b)
            k = pm.LogNormal(f"k_{ch}", np.log(k_median), k_sd)
            tau = pm.HalfNormal(f"tau_{ch}", 0.15)
            z = pm.Normal(f"region_z_{ch}", 0.0, 1.0, dims="region")
            mult = pt.exp(tau * z)
            ref = data.reference_activity(ch)
            sat = _adstock_hill(_lag_stack(data.drivers[ch]), alpha, k, ref)
            weighted = mult[:, None] * sat[:, :n]
            spend_fit = data.spend[ch][:, :n].sum() / Y_SCALE
            beta = roas * spend_fit * mult / (weighted.sum() + 1e-12)
            pm.Deterministic(f"beta_{ch}", beta, dims="region")
            contribution.append(beta[:, None] * sat)
            if calibrate and ch == "meta":
                _add_holdout(data, sat, beta, alpha, k, ref)
        contribution = pt.stack(contribution)  # (channel, region, week)
        pm.Deterministic(
            "channel_contribution", contribution.sum(axis=1) * Y_SCALE, dims=("channel", "week")
        )
        pm.Deterministic(
            "roas",
            pt.stack([model[f"roas_{ch}"] for ch in CHANNELS]),
            dims="channel",
        )

        mu = baseline + contribution.sum(axis=0)
        if store_mu:
            mu = pm.Deterministic("mu", mu, dims=("region", "week"))
        sigma = pm.HalfNormal("sigma", 0.1)
        # Noise in proportion to each region's size (its average sales), so a
        # large region is not held to a small region's error in pounds.
        size = np.exp(log_y_region)[:, None]
        pm.Normal("y", mu=mu[:, :n], sigma=sigma * size, observed=data.y[:, :n])
    return model


def _add_holdout(data: CaseData, sat, beta, alpha, k, ref) -> None:
    from mmm_book.experiment import EXPERIMENT_LIFT_GBP, EXPERIMENT_SE_GBP

    win = holdout_window()
    tr = win["treatment_regions"]
    w0 = win["start"]
    paused = data.drivers["meta"].copy()
    paused[tr, w0 : w0 + win["paused_weeks"]] = 0.0
    sat_paused = _adstock_hill(_lag_stack(paused), alpha, k, ref)
    window = slice(w0, w0 + win["measured_weeks"])
    lost = (beta[tr, None] * (sat[tr, window] - sat_paused[tr, window])).sum() * Y_SCALE
    implied = pm.Deterministic("holdout_implied_lift", lost)
    pm.Normal("holdout_lift", mu=implied, sigma=EXPERIMENT_SE_GBP, observed=EXPERIMENT_LIFT_GBP)


CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"


def _model_source() -> str:
    """Source of everything that affects a fit, for the cache key."""
    import inspect

    from mmm_book import experiment

    parts = [
        prepare_case_data,
        _panel,
        _lag_stack,
        _adstock_hill,
        holdout_window,
        build_case_model,
        _add_holdout,
    ]
    consts = repr((Y_SCALE, L_MAX, FOURIER_ORDER, SPEND_COL, TREATMENT_COL))
    consts += repr({k: v for k, v in vars(experiment).items() if k.startswith("EXPERIMENT_")})
    consts += inspect.getsource(experiment.select_pairs)
    return "".join(inspect.getsource(f) for f in parts) + consts + str(list(CONTROLS))


def fit_case_model(
    df: pd.DataFrame,
    variant: str = "model2",
    priors: CasePriors = PRIORS,
    calibrate: bool = False,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.97,
    random_seed: int = RANDOM_SEED,
    cache: bool = True,
    train_weeks: int | None = None,
) -> az.InferenceData:
    """Sample the case-study model, reusing a cached run when nothing has changed.

    The cache key covers the data, the variant, the priors, the calibration
    flag, the sampler settings and the source of every function that builds
    the model, so any change that could alter a fit forces a fresh one.
    Pass ``cache=False`` to always refit.
    """
    data = prepare_case_data(df, variant, train_weeks=train_weeks)
    key = hashlib.sha256(
        "|".join(
            [
                data.fingerprint(),
                priors.as_json(),
                str(calibrate),
                str(train_weeks),
                f"{draws}-{tune}-{chains}-{target_accept}-{random_seed}",
                _model_source(),
                pm.__version__,
            ]
        ).encode()
    ).hexdigest()[:20]
    tag = f"{variant}_{'cal' if calibrate else 'obs'}"
    if train_weeks:
        tag += f"_train{train_weeks}"
    path = CACHE_DIR / f"case_{tag}_{key}.nc"
    if cache and path.exists():
        return az.from_netcdf(path)
    model = build_case_model(data, priors, calibrate, store_mu=False)
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, 4),
            target_accept=target_accept,
            random_seed=random_seed,
            init="adapt_diag",
            progressbar=False,
        )
    if cache:
        CACHE_DIR.mkdir(exist_ok=True)
        idata.to_netcdf(path)
    return idata


def _draws(idata, name: str) -> np.ndarray:
    v = idata.posterior[name]
    return v.stack(sample=("chain", "draw")).transpose("sample", ...).to_numpy()


def roas_table(idata, data: CaseData, interval: float = 0.9) -> pd.DataFrame:
    """Posterior mean ROAS, interval and annual contribution per channel."""
    lo, hi = (1 - interval) / 2, 1 - (1 - interval) / 2
    rows = []
    for ch in CHANNELS:
        d = _draws(idata, f"roas_{ch}")
        rows.append(
            {
                "channel": ch,
                "roas_mean": d.mean(),
                "roas_lo": np.quantile(d, lo),
                "roas_hi": np.quantile(d, hi),
                "annual_contribution_m": d.mean() * data.annual_spend(ch) / 1e6,
            }
        )
    return pd.DataFrame(rows).set_index("channel")


def contribution_draws(idata, data: CaseData) -> dict[str, np.ndarray]:
    """Annual contribution draws per channel, in pounds."""
    return {ch: _draws(idata, f"roas_{ch}") * data.annual_spend(ch) for ch in CHANNELS}


def _hill_np(lags: np.ndarray, alpha: float, k: float, ref: np.ndarray) -> np.ndarray:
    w = alpha ** np.arange(L_MAX)
    w = w / w.sum()
    adstocked = np.tensordot(w, lags, axes=1)
    half = k * ref[:, None]
    return adstocked / (adstocked + half)


def _thin(n: int, max_draws: int | None) -> np.ndarray:
    """Evenly spaced draw indices, at most ``max_draws`` of them."""
    if max_draws is None or max_draws >= n:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_draws).round().astype(int))


def marginal_roas_draws(
    idata, data: CaseData, bump: float = 0.01, max_draws: int | None = None
) -> dict[str, np.ndarray]:
    """Posterior marginal ROAS: extra revenue from 1 per cent more spend, per pound.

    Raises each channel's driver by ``bump`` in every region-week, pushes it
    through each posterior draw's own adstock and saturation, and divides the
    extra revenue by the extra spend, as ``sedge.true_marginal_roas`` does
    for the truth. Every draw is used unless ``max_draws`` is given.
    """
    n = data.n_fit
    out = {}
    for ch in CHANNELS:
        alpha = _draws(idata, f"alpha_{ch}")
        k = _draws(idata, f"k_{ch}")
        beta = _draws(idata, f"beta_{ch}")
        x = data.drivers[ch]
        ref = data.reference_activity(ch)
        lags, lags_up = _lag_stack(x), _lag_stack(x * (1 + bump))
        extra_spend = data.spend[ch][:, :n].sum() * bump
        vals = []
        for i in _thin(len(alpha), max_draws):
            gain = _hill_np(lags_up, alpha[i], k[i], ref) - _hill_np(lags, alpha[i], k[i], ref)
            vals.append((beta[i][:, None] * gain[:, :n]).sum() * Y_SCALE / extra_spend)
        out[ch] = np.asarray(vals)
    return out


def holdout_lift_draws(idata, data: CaseData) -> np.ndarray:
    """Revenue the Chapter 36 Meta pause costs, per posterior draw, in pounds.

    The same calculation as the calibration term: Meta switched off in the
    treatment regions for the six test weeks, pushed through each draw's
    adstock and saturation, and summed over the test weeks plus the
    cooldown. It works for any fit, so the uncalibrated model's prediction
    of the experiment can be compared with what the experiment measured.
    """
    win = holdout_window()
    tr = win["treatment_regions"]
    w0 = win["start"]
    window = slice(w0, w0 + win["measured_weeks"])
    x = data.drivers["meta"]
    paused = x.copy()
    paused[tr, w0 : w0 + win["paused_weeks"]] = 0.0
    lags, lags_paused = _lag_stack(x), _lag_stack(paused)
    ref = data.reference_activity("meta")
    alpha = _draws(idata, "alpha_meta")
    k = _draws(idata, "k_meta")
    beta = _draws(idata, "beta_meta")
    out = np.empty(len(alpha))
    for i in range(len(alpha)):
        lost = _hill_np(lags, alpha[i], k[i], ref) - _hill_np(lags_paused, alpha[i], k[i], ref)
        out[i] = (beta[i][tr, None] * lost[tr, window]).sum() * Y_SCALE
    return out


def posterior_curves(idata, data: CaseData, max_draws: int | None = 2000):
    """Response curves built from every posterior draw's own fitted parameters.

    Returns an ``optimise.PosteriorCurves``: for each draw, each channel's
    national weekly contribution as a function of its average weekly spend,
    using that draw's adstock retention, half-saturation point and regional
    scales. A budget change scales the channel's whole historical plan in
    every region and week, as ``optimise.ResponseCurves`` does.
    """
    from mmm_book.optimise import PosteriorCurves

    n = data.n_fit
    adstocked, half, beta = {}, {}, {}
    for ch in CHANNELS:
        alpha = _draws(idata, f"alpha_{ch}")
        idx = _thin(len(alpha), max_draws)
        lags = _lag_stack(data.drivers[ch])[:, :, :n]
        a = alpha[idx]
        w = a[:, None] ** np.arange(L_MAX)[None, :]
        w = w / w.sum(axis=1, keepdims=True)
        adstocked[ch] = np.einsum("dl,lrt->drt", w, lags)
        half[ch] = _draws(idata, f"k_{ch}")[idx][:, None] * data.reference_activity(ch)[None, :]
        beta[ch] = _draws(idata, f"beta_{ch}")[idx] * Y_SCALE
    current = {ch: float(data.spend[ch][:, :n].sum()) / n for ch in CHANNELS}
    return PosteriorCurves(adstocked, half, beta, current)


def posterior_mu(
    idata,
    data: CaseData,
    priors: CasePriors = PRIORS,
    calibrate: bool = False,
    thin: int = 10,
) -> np.ndarray:
    """Posterior mean of expected revenue (region x week, model units), from every ``thin``-th draw.

    Pass the ``data``, ``priors`` and ``calibrate`` the fit used.
    """
    model = build_case_model(data, priors, calibrate, store_mu=True)
    post = idata.posterior.isel(draw=slice(None, None, thin))
    mu = pm.compute_deterministics(post, model=model, var_names=["mu"], progressbar=False)
    return mu["mu"].mean(("chain", "draw")).to_numpy()


# Reference levels for the decomposition of the controls into pounds: no
# promotion and the list price, with every other control at its average.
# Each is in the units of ``CONTROLS`` (logs where the control enters as a log).
def control_reference(data: CaseData) -> dict[str, float]:
    from mmm_book.sedge import PRICE_LIST

    names = data.control_names
    ref = dict(zip(names, data.control_centre, strict=True))
    ref["price"] = float(np.log(PRICE_LIST))
    ref["promo_depth"] = 0.0
    return ref


def decompose(idata, data: CaseData, max_draws: int | None = 500) -> dict[str, np.ndarray]:
    """Annual revenue split into baseline, controls and media, per posterior draw (pounds).

    The case model's controls act multiplicatively on the baseline, so their
    pound contribution is the baseline times (multiplier - 1), with the
    multiplier measured from each control's reference level
    (``control_reference``: no promotion, list price, everything else at its
    average). The baseline is the regional level with trend, seasonality and
    launches, at the reference controls; the media are the five channels'
    contributions. ``promotion`` and ``price`` are those two controls' own
    effects with the others at reference; ``controls`` is all seven together.
    """
    names = data.control_names
    ref = control_reference(data)
    z_ref = np.array(
        [
            (ref[c] - m) / sd
            for c, m, sd in zip(names, data.control_centre, data.control_scale, strict=True)
        ]
    )
    level = _draws(idata, "level")
    trend = _draws(idata, "trend")
    season = _draws(idata, "season")
    launch = _draws(idata, "launch_effect")
    gamma = _draws(idata, "gamma")
    idx = _thin(len(level), max_draws)
    years = data.n_years
    out = {key: [] for key in ("baseline", "controls", "promotion", "price", "media", "fitted")}
    # Summing beta x saturation over regions and weeks gives ROAS x spend exactly.
    media_total = sum(_draws(idata, f"roas_{ch}") * float(data.spend[ch].sum()) for ch in CHANNELS)
    p, q = names.index("promo_depth"), names.index("price")
    for i in idx:
        log_base = (
            trend[i] * data.trend[None, :]
            + (data.fourier @ season[i])[None, :]
            + (data.events @ launch[i])[None, :]
        )
        base = level[i][:, None] * np.exp(log_base + (z_ref @ gamma[i]))
        dz = data.controls - z_ref
        mult_all = np.exp(dz @ gamma[i])
        mult_promo = np.exp(dz[..., p] * gamma[i][p])
        mult_price = np.exp(dz[..., q] * gamma[i][q])
        media = media_total[i]
        out["baseline"].append(base.sum() * Y_SCALE / years)
        out["controls"].append((base * (mult_all - 1)).sum() * Y_SCALE / years)
        out["promotion"].append((base * (mult_promo - 1)).sum() * Y_SCALE / years)
        out["price"].append((base * (mult_price - 1)).sum() * Y_SCALE / years)
        out["media"].append(media / years)
        out["fitted"].append((base * mult_all).sum() * Y_SCALE / years + media / years)
    result = {key: np.asarray(v) for key, v in out.items()}
    result["actual"] = np.full(len(idx), data.y.sum() * Y_SCALE / years)
    result["residual"] = result["actual"] - result["fitted"]
    return result


def fit_summary(mu: np.ndarray, data: CaseData) -> dict[str, float]:
    """In-sample R squared and revenue-weighted MAPE, given ``posterior_mu``."""
    y = data.y
    r2 = 1 - ((y - mu) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    wmape = np.abs(y - mu).sum() / y.sum()
    return {"r2": float(r2), "wmape": float(wmape)}


def residuals(mu: np.ndarray, data: CaseData) -> pd.DataFrame:
    """Residuals (actual minus ``posterior_mu``) as a region x week table, in pounds."""
    return pd.DataFrame((data.y - mu) * Y_SCALE, index=data.regions, columns=data.weeks)
