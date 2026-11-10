"""Every number Part IX of the book prints, computed from the case-study fits.

The Sedge Shoes chapters (34 to 39) report what the fitted case-study model
says, with intervals, and Appendix F compares it with the true values the
data generator built in. This module is the single place those numbers come
from:

* ``case_fits`` fits (or loads from the cache) the five case-study models:
  Model 1, Model 2, Model 2 trained on 130 and on 104 weeks, and Model 2
  calibrated to the Meta holdout.
* ``part9_results`` turns them into a nested dictionary of every printed
  figure, chapter by chapter.
* ``write_results`` saves that dictionary as JSON (``results/part9.json``),
  which the book's figures read, so a figure and its chapter cannot drift
  apart.

The notebooks in ``notebooks/case_study`` print the same quantities step by
step, and ``tests/test_book_part9.py`` pins the values the book prints.

Run ``python -m mmm_book.case_results`` to refresh the JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy import stats

from mmm_book import RANDOM_SEED
from mmm_book.case_model import (
    FIRST_PRIORS,
    Y_SCALE,
    contribution_draws,
    decompose,
    fit_case_model,
    fit_summary,
    holdout_lift_draws,
    marginal_roas_draws,
    posterior_curves,
    posterior_mu,
    prepare_case_data,
)
from mmm_book.experiment import (
    EXPERIMENT_LIFT_GBP,
    EXPERIMENT_SE_GBP,
    EXPERIMENT_WITHHELD_SPEND_GBP,
)
from mmm_book.optimise import (
    BOOK_PLAN_ANNUAL,
    band,
    book_plan_bounds,
    greedy_allocate,
)
from mmm_book.sedge import (
    CHANNELS,
    GROUND_TRUTH,
    LAUNCHES,
    correct_distribution,
    generate_sedge,
    true_marginal_roas,
)

WEEKS = 52
MARGIN = 0.5  # Sedge's contribution margin on incremental revenue (Chapter 37)
BREAK_EVEN = 1 / MARGIN  # a pound of media pays for itself at 2.0x
INTERVAL = (0.05, 0.95)  # 90 per cent intervals throughout
RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "part9.json"

# Chapter 27's two plans (annual GBP m), built on the teaching curves and
# scored here on the fitted model so that Chapter 38 can compare like with like.
CH27_PLANS = {
    "ch27_unconstrained": {"search": 1.19, "meta": 0.68, "youtube": 1.13, "tv": 3.00, "ooh": 0.00},
    "ch27_constrained": {"search": 1.22, "meta": 1.20, "youtube": 1.04, "tv": 1.82, "ooh": 0.72},
}


def _q(x: np.ndarray) -> list[float]:
    return [float(np.quantile(x, INTERVAL[0])), float(np.quantile(x, INTERVAL[1]))]


def _summ(x: np.ndarray, scale: float = 1.0) -> dict:
    x = np.asarray(x) * scale
    lo, hi = _q(x)
    return {"mean": float(x.mean()), "lo": lo, "hi": hi}


def _diagnostics(idata) -> dict:
    # Free parameters only: deterministics are functions of them.
    skip = ("beta_", "channel_contribution", "holdout_implied")
    names = [
        v
        for v in idata.posterior.data_vars
        if not v.startswith(skip) and v not in ("roas", "level", "mu")
    ]
    s = az.summary(idata, var_names=names, kind="diagnostics")
    return {
        "divergences": int(idata.sample_stats["diverging"].sum()),
        "max_rhat": float(s["r_hat"].max()),
        "min_ess_bulk": float(s["ess_bulk"].min()),
        "min_ess_tail": float(s["ess_tail"].min()),
        "draws": int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]),
    }


def _roas(idata, ch: str) -> np.ndarray:
    return idata.posterior[f"roas_{ch}"].to_numpy().ravel()


def _roas_block(idata, data) -> dict:
    out = {}
    for ch in CHANNELS:
        r = _roas(idata, ch)
        out[ch] = {
            **_summ(r),
            "contribution_m": float(r.mean() * data.annual_spend(ch) / 1e6),
            "spend_m": data.annual_spend(ch) / 1e6,
        }
    total_c = sum(v["contribution_m"] for v in out.values())
    total_s = sum(v["spend_m"] for v in out.values())
    out["total"] = {"contribution_m": total_c, "spend_m": total_s, "roas": total_c / total_s}
    return out


def case_fits(df: pd.DataFrame | None = None) -> dict:
    """Fit (or load) the five case-study models on the corrected Sedge panel."""
    fixed = correct_distribution(generate_sedge(seed=RANDOM_SEED)) if df is None else df
    fits = {}
    for key, variant, calibrate, train in [
        ("model1", "model1", False, None),
        ("model2", "model2", False, None),
        ("model2_cal", "model2", True, None),
        ("model2_train130", "model2", False, 130),
        ("model2_train104", "model2", False, 104),
    ]:
        idata = fit_case_model(fixed, variant, calibrate=calibrate, train_weeks=train)
        data = prepare_case_data(fixed, variant, train_weeks=train)
        fits[key] = (idata, data, calibrate)
    fits["panel"] = fixed
    return fits


# ---- Chapter 34 ------------------------------------------------------------


def _corr(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def chapter34(raw: pd.DataFrame) -> dict:
    nat = raw.groupby("week").sum(numeric_only=True)
    cat = raw.groupby("week")["category_search"].mean()
    years = raw["week"].nunique() / 52
    return {
        "rows": len(raw),
        "regions": int(raw["region"].nunique()),
        "weeks": int(raw["week"].nunique()),
        "corr_meta_youtube_national": _corr(nat["meta_spend"], nat["youtube_spend"]),
        "corr_meta_youtube_panel": _corr(raw["meta_spend"], raw["youtube_spend"]),
        "corr_search_category": _corr(nat["search_spend"], cat),
        "annual_revenue_m": float(raw["revenue"].sum() / years / 1e6),
    }


# ---- Chapter 35 ------------------------------------------------------------


def chapter35(fits: dict) -> dict:
    idata1, data1, _ = fits["model1"]
    idata2, data2, _ = fits["model2"]
    # Prior predictive, exactly: the share of the prior above a threshold.
    median, sd = FIRST_PRIORS.roas["ooh"]
    ooh_threshold = 20e6 / data2.annual_spend("ooh")
    a, b = FIRST_PRIORS.adstock["search"]
    alpha_8wk = 0.5 ** (1 / 8)  # retention above this means a half-life beyond eight weeks
    fit1 = fit_summary(posterior_mu(idata1, data1), data1)
    fit2 = fit_summary(posterior_mu(idata2, data2), data2)
    contrib = contribution_draws(idata2, data2)
    meta, yt = contrib["meta"] / 1e6, contrib["youtube"] / 1e6
    return {
        "prior_ooh_share_above_20m": float(stats.lognorm(sd, scale=median).sf(ooh_threshold)),
        "prior_search_share_halflife_above_8": float(stats.beta(a, b).sf(alpha_8wk)),
        "model1": {
            "roas": _roas_block(idata1, data1),
            "r2": fit1["r2"],
            "wmape": fit1["wmape"],
            "diagnostics": _diagnostics(idata1),
        },
        "model2": {
            "roas": _roas_block(idata2, data2),
            "r2": fit2["r2"],
            "wmape": fit2["wmape"],
            "diagnostics": _diagnostics(idata2),
        },
        "meta_m": _summ(meta),
        "youtube_m": _summ(yt),
        "pair_m": _summ(meta + yt),
        "pair_corr": float(np.corrcoef(meta, yt)[0, 1]),
        # every fifth draw, for Figure 35.3's scatter
        "pair_draws_m": [[float(m), float(y)] for m, y in zip(meta[::5], yt[::5], strict=True)],
    }


# ---- Chapter 36 ------------------------------------------------------------


def _validation(fits: dict) -> dict:
    idata, data, _ = fits["model2_train130"]
    n = data.n_fit
    mu = posterior_mu(idata, data) * Y_SCALE
    actual = data.y.sum(axis=0) * Y_SCALE
    pred = mu.sum(axis=0)
    test = data.weeks >= n
    start = LAUNCHES[2][0]
    launch = (data.weeks >= start) & (data.weeks < start + 6)
    err = actual - pred
    return {
        "train_weeks": n,
        "test_wmape": float(np.abs(err)[test].sum() / actual[test].sum()),
        "mean_error_outside_launch": float(err[test & ~launch].mean()),
        "mean_error_in_launch": float(err[launch].mean()),
        "launch_week": start,
        "diagnostics": _diagnostics(idata),
        "diagnostics_104": _diagnostics(fits["model2_train104"][0]),
    }


def chapter36(fits: dict) -> dict:
    idata2, data2, _ = fits["model2"]
    idatac, datac, _ = fits["model2_cal"]
    folds = {}
    for key, label in [("model2_train104", "104"), ("model2_train130", "130"), ("model2", "156")]:
        idata = fits[key][0]
        folds[label] = {ch: float(_roas(idata, ch).mean()) for ch in CHANNELS}
    lift_before = holdout_lift_draws(idata2, data2)
    lift_after = holdout_lift_draws(idatac, datac)
    meta_b, meta_a = _roas(idata2, "meta"), _roas(idatac, "meta")
    yt_b, yt_a = _roas(idata2, "youtube"), _roas(idatac, "youtube")
    # How precisely does each source pin the test's lift? The uncalibrated
    # model's posterior for the lift, against the experiment's own standard error.
    sd_model = float(lift_before.std())
    combined_sd = 1 / np.sqrt(1 / sd_model**2 + 1 / EXPERIMENT_SE_GBP**2)
    dec = decompose(idatac, datac)
    final = _roas_block(idatac, datac)
    profit = {ch: MARGIN * final[ch]["contribution_m"] - final[ch]["spend_m"] for ch in CHANNELS}
    truth = {
        ch: {
            "roas": GROUND_TRUTH[ch],
            "contribution_m": GROUND_TRUTH[ch] * datac.annual_spend(ch) / 1e6,
            "inside_interval": bool(final[ch]["lo"] <= GROUND_TRUTH[ch] <= final[ch]["hi"]),
        }
        for ch in CHANNELS
    }
    return {
        "validation": _validation(fits),
        "folds": folds,
        "model2": _roas_block(idata2, data2),
        "holdout": {
            "observed_lift": EXPERIMENT_LIFT_GBP,
            "se": EXPERIMENT_SE_GBP,
            "withheld": EXPERIMENT_WITHHELD_SPEND_GBP,
            "lift_before": _summ(lift_before),
            "lift_after": _summ(lift_after),
            "lift_before_sd": sd_model,
            "lift_after_sd": float(lift_after.std()),
            "combined_sd_if_independent": float(combined_sd),
            "test_roas_before": _summ(lift_before / EXPERIMENT_WITHHELD_SPEND_GBP),
            "test_roas_after": _summ(lift_after / EXPERIMENT_WITHHELD_SPEND_GBP),
            "annual_over_test_roas_before": _summ(
                meta_b / (lift_before / EXPERIMENT_WITHHELD_SPEND_GBP)
            ),
            "meta_before": _summ(meta_b),
            "meta_after": _summ(meta_a),
            # Appendix H's interval figure: chance the calibrated Meta ROAS clears break-even
            "meta_after_p_above_break_even": float((meta_a > BREAK_EVEN).mean()),
            "meta_width_before": float(np.subtract(*_q(meta_b)[::-1])),
            "meta_width_after": float(np.subtract(*_q(meta_a)[::-1])),
            "meta_sd_before": float(meta_b.std()),
            "meta_sd_after": float(meta_a.std()),
            "youtube_before": _summ(yt_b),
            "youtube_after": _summ(yt_a),
            "corr_lift_meta_before": float(np.corrcoef(lift_before, meta_b)[0, 1]),
            "diagnostics": _diagnostics(idatac),
        },
        "final": final,
        "profit_m": {**profit, "total": sum(profit.values())},
        "truth": truth,
        "decomposition_m": {k: _summ(v / 1e6) for k, v in dec.items()},
        "meta_density": _density(meta_b, meta_a),
    }


def _density(*samples, lo: float = 0.0, hi: float = 4.0, n: int = 161) -> dict:
    """Kernel density of each sample on a common grid, for the calibration figure."""
    grid = np.linspace(lo, hi, n)
    return {
        "grid": grid.tolist(),
        "densities": [stats.gaussian_kde(x)(grid).tolist() for x in samples],
    }


# ---- Chapters 37 to 39 -------------------------------------------------------


def _annual(weekly: dict) -> dict:
    return {ch: weekly[ch] * WEEKS / 1e6 for ch in CHANNELS}


def _weekly(annual_m: dict, curves) -> dict:
    scale = curves.total_budget * WEEKS / 6e6  # the panel spends exactly 6.0m a year
    return {ch: annual_m[ch] * 1e6 / WEEKS * scale for ch in CHANNELS}


def _score(curves, plan_weekly: dict, base: dict | None = None) -> dict:
    gain = curves.gain_draws(plan_weekly, base) * WEEKS / 1e6
    return {
        "plan_m": _annual(plan_weekly),
        "gain_m": _summ(gain),
        "p_gain": float((gain > 0).mean()),
        "profit_m": float(MARGIN * gain.mean()),
    }


def chapter37(fits: dict, curves) -> dict:
    idatac, datac, _ = fits["model2_cal"]
    marg = marginal_roas_draws(idatac, datac)
    true_marg = true_marginal_roas()
    out = {}
    for ch in CHANNELS:
        r = _roas(idatac, ch)
        out[ch] = {
            "average": _summ(r),
            "marginal": _summ(marg[ch]),
            "p_marginal_above_break_even": float((marg[ch] > BREAK_EVEN).mean()),
            "marginal_from_curves": float(curves.marginal_return(ch)),
            "true_average": GROUND_TRUTH[ch],
            "true_marginal": true_marg[ch],
        }
    return out


def _bounded_plan(curves) -> dict:
    lo, hi = book_plan_bounds(curves)
    return greedy_allocate(curves, lo, hi)


def chapter38(fits: dict, curves) -> dict:
    current = curves.current_money
    lo40, hi40 = band(curves, 0.40)
    naive = greedy_allocate(curves, lo40, hi40)
    optimiser = _bounded_plan(curves)
    plan = _weekly({ch: BOOK_PLAN_ANNUAL[ch] / 1e6 for ch in CHANNELS}, curves)
    plans = {
        "naive_40": _score(curves, naive),
        "bounded_optimiser": _score(curves, optimiser),
        "operational": _score(curves, plan),
    }
    for key, annual in CH27_PLANS.items():
        plans[key] = _score(curves, _weekly(annual, curves))
    base_rev = curves.gain_draws(plan, {ch: 0.0 for ch in CHANNELS}) * WEEKS / 1e6
    current_rev = curves.gain_draws(current, {ch: 0.0 for ch in CHANNELS}) * WEEKS / 1e6
    scenarios = {}
    for label, factor in [("cut15", 0.85), ("up10", 1.10)]:
        scaled = {ch: plan[ch] * factor for ch in CHANNELS}
        rev = curves.gain_draws(scaled, {ch: 0.0 for ch in CHANNELS}) * WEEKS / 1e6
        diff = rev - base_rev
        spend_change = (factor - 1) * 6.0
        scenarios[label] = {
            "budget_m": 6.0 * factor,
            "revenue_m": _summ(rev),
            "difference_m": _summ(diff),
            "marginal_roas": float(diff.mean() / spend_change),
            "profit_change_m": float(MARGIN * diff.mean() - spend_change),
            "p_profit_up": float((MARGIN * diff - spend_change > 0).mean()),
        }
    lo, hi = band(curves, 0.40)
    targeted = greedy_allocate(curves, lo, hi, total=curves.total_budget * 5.1 / 6.0)
    t_rev = curves.gain_draws(targeted, {ch: 0.0 for ch in CHANNELS}) * WEEKS / 1e6
    scenarios["targeted_cut"] = {
        "plan_m": _annual(targeted),
        "difference_m": _summ(t_rev - base_rev),
        "profit_change_m": float(MARGIN * (t_rev - base_rev).mean() + 0.9),
    }
    gain = curves.gain_draws(plan) * WEEKS / 1e6
    bins = np.linspace(np.floor(gain.min() * 20) / 20, np.ceil(gain.max() * 20) / 20, 41)
    counts, _ = np.histogram(gain, bins=bins)
    bands = {}
    for ch in CHANNELS:
        grid = np.linspace(0.0, 2.6e6, 66)  # annual spend, 0 to 2.6m
        draws = np.array([curves.draws(ch, x / WEEKS) for x in grid]) * WEEKS / 1e6
        lo_b, hi_b = np.quantile(draws, INTERVAL, axis=1)
        bands[ch] = {
            "spend_m": (grid / 1e6).tolist(),
            "mean_m": draws.mean(axis=1).tolist(),
            "lo_m": lo_b.tolist(),
            "hi_m": hi_b.tolist(),
            "current_spend_m": current[ch] * WEEKS / 1e6,
            "current_revenue_m": float(curves.money_contribution(ch, current[ch]) * WEEKS / 1e6),
        }
    # The naive plan leans on YouTube: its next pound across the curve draws.
    youtube_marginal = _summ(curves.marginal_draws("youtube"))
    # Why the operational plan's direction is near certain: in how many draws
    # does Television's next pound beat Meta's?
    p_tv_beats_meta = float((curves.marginal_draws("tv") > curves.marginal_draws("meta")).mean())
    # Inside the fixed budget: what the plan's last 50,000 earns in each channel
    # it moves (Meta's 50,000 kept, YouTube's and OOH's 50,000 added).
    steps = {}
    for ch, a, b in [
        ("meta", 1.25e6, 1.30e6),
        ("youtube", 0.80e6, 0.85e6),
        ("ooh", 0.90e6, 0.95e6),
    ]:
        extra = (
            curves.money_contribution(ch, b / WEEKS) - curves.money_contribution(ch, a / WEEKS)
        ) * WEEKS
        steps[ch] = {
            "from_m": a / 1e6,
            "to_m": b / 1e6,
            "revenue": float(extra),
            "per_pound": float(extra / (b - a)),
            "profit": float(MARGIN * extra - (b - a)),
        }
    return {
        "curve_bands": bands,
        "youtube_marginal": youtube_marginal,
        "p_tv_marginal_above_meta": p_tv_beats_meta,
        "fixed_budget_steps": steps,
        "operational_gain_hist": {"edges_m": bins.tolist(), "counts": counts.tolist()},
        "current_revenue_m": _summ(current_rev),
        "operational_revenue_m": _summ(base_rev),
        "plans": plans,
        "scenarios": scenarios,
        "n_draws": int(curves.n_draws),
    }


def chapter39(fits: dict, curves, ch38: dict) -> dict:
    """Value of information: rerun the optimiser at each channel's p10 and p90 marginal return.

    For each channel, the draws are split by that channel's marginal return,
    and the optimiser is rerun on the mean curves of the bottom tenth and of
    the top tenth (every channel's curve from those draws, so the other
    channels move with it as the posterior says they should). The channel
    whose uncertainty moves the plan furthest is the most valuable to test.
    """
    idatac, datac, _ = fits["model2_cal"]
    marg = {ch: curves.marginal_draws(ch) for ch in CHANNELS}
    subsets = {}
    for ch in CHANNELS:
        lo_cut, hi_cut = np.quantile(marg[ch], [0.10, 0.90])
        subsets[ch] = (
            curves.subset(marg[ch] <= lo_cut),
            curves.subset(marg[ch] >= hi_cut),
        )
    lo40, hi40 = band(curves, 0.20)
    voi = {}
    for ch, (low, high) in subsets.items():
        a = greedy_allocate(low, lo40, hi40)
        b = greedy_allocate(high, lo40, hi40)
        moved = sum(abs(a[c] - b[c]) for c in CHANNELS) * WEEKS / 1e6 / 2
        voi[ch] = {
            "marginal_p10": float(np.quantile(marg[ch], 0.10)),
            "marginal_p90": float(np.quantile(marg[ch], 0.90)),
            "low_plan_m": _annual(a),
            "high_plan_m": _annual(b),
            "moved_m": float(moved),
        }
    ranking = sorted(CHANNELS, key=lambda c: -voi[c]["moved_m"])
    op = ch38["plans"]["operational"]
    return {
        "voi": voi,
        "voi_ranking": ranking,
        "half_year_gain_m": op["gain_m"]["mean"] / 2,
        "half_year_profit_m": op["profit_m"] / 2,
    }


def chapter27(curves, n_resolve: int = 300) -> dict:
    """Chapter 27's constrained plan scored, and re-solved, on the fitted model's draws.

    Chapter 27 builds its plans on the teaching curves; its uncertainty
    statements come from here: the plan's gain on every posterior draw of
    the calibrated case-study model, and the plan re-solved on each of
    ``n_resolve`` individual draws within Chapter 27's bounds (plus or minus
    20 per cent for Search, Meta and OOH, 30 per cent for YouTube and
    Television).
    """
    plan = _weekly(CH27_PLANS["ch27_constrained"], curves)
    gain = curves.gain_draws(plan) * WEEKS / 1e6
    bins = np.linspace(np.floor(gain.min() * 20) / 20, np.ceil(gain.max() * 20) / 20, 41)
    counts, _ = np.histogram(gain, bins=bins)
    lo, hi = band(curves, 0.20)
    for ch in ("youtube", "tv"):
        lo[ch] = curves.current_money[ch] * 0.70
        hi[ch] = curves.current_money[ch] * 1.30
    idx = np.unique(np.linspace(0, curves.n_draws - 1, n_resolve).round().astype(int))
    allocs = {ch: [] for ch in CHANNELS}
    for i in idx:
        keep = np.zeros(curves.n_draws, dtype=bool)
        keep[i] = True
        a = greedy_allocate(curves.subset(keep), lo, hi, n_steps=600)
        for ch in CHANNELS:
            allocs[ch].append(a[ch] * WEEKS / 1e6)
    pct = [5, 25, 50, 75, 95]
    return {
        "gain_m": _summ(gain),
        "p_gain": float((gain > 0).mean()),
        "p_gain_below_100k": float((gain < 0.1).mean()),
        "gain_hist": {"edges_m": bins.tolist(), "counts": counts.tolist()},
        "resolved_percentiles_m": {ch: np.percentile(allocs[ch], pct).tolist() for ch in CHANNELS},
        "bounds_m": {ch: [lo[ch] * WEEKS / 1e6, hi[ch] * WEEKS / 1e6] for ch in CHANNELS},
        "n_resolved": len(idx),
        "n_draws": int(curves.n_draws),
    }


def part9_results(fits: dict | None = None, max_curve_draws: int | None = 2000) -> dict:
    """Every number Part IX prints, chapter by chapter."""
    fits = case_fits() if fits is None else fits
    raw = generate_sedge(seed=RANDOM_SEED)
    idatac, datac, _ = fits["model2_cal"]
    curves = posterior_curves(idatac, datac, max_draws=max_curve_draws)
    ch38 = chapter38(fits, curves)
    return {
        "ch27": chapter27(curves),
        "ch34": chapter34(raw),
        "ch35": chapter35(fits),
        "ch36": chapter36(fits),
        "ch37": chapter37(fits, curves),
        "ch38": ch38,
        "ch39": chapter39(fits, curves, ch38),
    }


def write_results(path: Path = RESULTS_PATH, results: dict | None = None) -> Path:
    results = part9_results() if results is None else results
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    return path


if __name__ == "__main__":
    print(write_results())
