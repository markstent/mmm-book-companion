"""Synthetic "Sedge Shoes" dataset generator for the MMM book case study.

Sedge is a made-up UK running-shoe retailer selling across 18 regions.
The generator builds the weekly region panel that Chapter 34 describes,
with the same columns, the same awkward correlations and the same traps:

* Television is bought in flights around the spring races and Christmas,
  so it moves with seasonality and with promotions.
* Meta and YouTube are planned from one paid-social budget, so they move
  together (about 0.85 nationally, about 0.76 across the region-weeks).
* Paid Search spend follows category demand (about 0.72 with the category
  search index), and part of it is brand search, which captures demand that
  already exists and that television, Meta and YouTube create.
* Revenue also responds to something the data only partly record: a
  market-wide demand shock (the running calendar, the weather on race days)
  that the category search index proxies imperfectly. There is no
  competitor activity in the data.
* In one region a new retail partner went live but the distribution series
  was updated three months late (``RETAILER_ONBOARDING``), the data problem
  Chapter 35's baseline model finds. ``correct_distribution`` repairs it.

Revenue in each region-week is an organic baseline (level, trend,
seasonality and launches) scaled by price and promotion depth (non-media
treatments), the controls (distribution, stock, weather) and the demand
shock, all multiplicative,
plus a contribution from each marketing channel. Each channel's driver
(GRPs for television, generic search impressions for Paid Search, spend
for the rest) is adstocked with
``geometric_adstock`` and saturated with ``hill_saturation``, then scaled
so that the channel's aggregate ROAS matches ``GROUND_TRUTH`` *by
construction*:

    sum(contribution_c) / sum(spend_c) == GROUND_TRUTH[c]

holds exactly for the generated dataset. Brand search spend has no causal
effect of its own: its revenue is demand it collects, which is why
Chapter 35 takes it out of the media set. Paid Search's ROAS is still
quoted on total Search spend, brand included, as the book does.

``ground_truth_contributions`` returns the by-construction per-channel
contributions (row-aligned with ``generate_sedge`` for the same arguments)
and ``true_marginal_roas`` the marginal returns at actual spend, so labs and
tests can check a recovered ROAS against the known answer.

Everything is drawn from one ``numpy.random.default_rng(seed)`` stream, so
a seed always gives the same panel.
"""

import numpy as np
import pandas as pd

from mmm_book import RANDOM_SEED
from mmm_book.adstock import geometric_adstock
from mmm_book.saturation import hill_saturation

CHANNELS = ["tv", "search", "meta", "youtube", "ooh"]

# True average ROAS per channel, the book's canonical Sedge ledger:
# Television 2.4x, Paid Search 2.7x, Meta 1.75x, YouTube 2.0x, OOH 1.5x.
# The chapters before Part IX use them as Sedge's working numbers, and
# Chapter 36 checks the calibrated model against them. A test
# (tests/test_book_ledger.py) ties these values to the book.
GROUND_TRUTH = {"tv": 2.4, "search": 2.7, "meta": 1.75, "youtube": 2.0, "ooh": 1.5}

# Annual media spend per channel in the book (GBP), Chapter 34. Total GBP 6.0m.
# Each channel's plan is scaled so the panel spends exactly this a year.
BOOK_ANNUAL_SPEND = {
    "tv": 1_400_000.0,
    "search": 1_400_000.0,
    "meta": 1_500_000.0,
    "youtube": 800_000.0,
    "ooh": 900_000.0,
}
BOOK_N_REGIONS = 18

_START_DATE = pd.Timestamp("2022-01-03")

# Per-channel adstock retention and Hill slope used to generate the data.
# They are public so labs can rebuild correctly specified features and so
# the case-study model's estimates can be compared with the truth.
ADSTOCK_ALPHA = {"tv": 0.5, "search": 0.3, "meta": 0.4, "youtube": 0.45, "ooh": 0.6}
HILL_SLOPE = {"tv": 1.0, "search": 1.1, "meta": 1.0, "youtube": 1.0, "ooh": 1.0}

# Half-saturation point as a multiple of each region's average adstocked
# driver (its average weekly activity). Television is far from saturation at its operating point
# (marginal ROAS close to average, book 2.1x against 2.4x); Paid Search is
# deep in its curve (book marginal 1.4x against 2.7x average). The
# multiples are solved so that ``true_marginal_roas`` reproduces the book's
# marginal ledger (tests/test_book_ledger.py checks it).
HALF_MULT = {"tv": 14.47, "search": 0.923, "meta": 1.071, "youtube": 9.58, "ooh": 50.0}

# Television is bought in GRPs; this converts GRPs to pounds, and the
# ``tv_spend`` column is exactly ``tv_grps * TV_COST_PER_GRP``.
TV_COST_PER_GRP = 400.0

# Calendar (week of year, 0 = first week of January).
_RACE_WEEKS = range(10, 17)  # mid-March to late April: the spring marathons
_XMAS_WEEKS = range(44, 51)  # November to mid-December: Black Friday and Christmas
# Three product launches, six weeks each: two smaller ones (March 2023 and
# March 2024) and a materially larger one in week 141 (September 2024), the
# launch Chapter 36's holdout misses. Uplift is the share of baseline revenue.
LAUNCHES = [(62, 0.08), (114, 0.08), (141, 0.25)]
LAUNCH_LENGTH = 6

# The distribution data problem of Chapter 35: a new retail partner goes
# live in one region in ``week`` but the distribution series records it
# ``lag_weeks`` late. The region also gets an OOH push for the opening.
RETAILER_ONBOARDING = {"region": "region_11", "week": 70, "lag_weeks": 12, "step": 0.10}
_RETAILER_OOH_WEEKS = 16

# Chapter 35's past search-pause test, which is in the data: generic search
# off in six regions for eight weeks in summer 2022, brand search left on.
SEARCH_PAUSE = {"regions": [3, 5, 8, 12, 14, 16], "weeks": range(30, 38)}
_RETAILER_OOH_MULT = 2.5

# Weeks in which the paid-social plan is held steady: the four weeks
# before the Chapter 36 Meta holdout (weeks 148 to 153) and the test and
# cooldown themselves.
_STEADY_SOCIAL_WEEKS = range(144, 156)

# Organic baseline demand (GBP a week per region before the multiplicative
# effects below).
_BASE_LEVEL = 39_950.0
_TREND_PER_WEEK = 12.0
_SEASONAL_AMPLITUDE = 0.18

# Average selling price, GBP a pair, before promotional discount.
PRICE_LIST = 95.0
_PRICE_NOISE = 0.03
_PRICE_ELASTICITY = 1.3
# Share of a promotion's depth that shows up in the week's average price
# (not every pair sold that week is on promotion).
_PROMO_PRICE_PASS_THROUGH = 0.5
_PROMO_UPLIFT_PER_DEPTH = 0.8

_DISTRIBUTION_MEAN = 0.85
_DISTRIBUTION_STD = 0.03

_STOCK_MEAN = 0.97
_STOCK_STD = 0.008
_STOCKOUT_WEEKS = range(84, 92)  # a supply problem in August and September 2023
_STOCKOUT_DEPTH = 0.10

_TEMP_EFFECT = 0.008  # per degree C above 10
_RAIN_EFFECT = -0.004  # per mm a week above 15

# Unobserved demand: a national AR(1) market-wide demand shock (the running
# calendar, the weather on race day) and a smaller regional one. The data
# contain no competitor activity.
_SHOCK_PHI = 0.85
_SHOCK_NAT_SD = 0.035
_SHOCK_REG_SD = 0.012
# The category search index sees most, not all, of the national shock.
_CATEGORY_SHOCK_LOADING = 0.6
# Category seasonality is real demand: it moves sales as well as searches.
_CATEGORY_SEASON_EFFECT = 0.5

_NOISE_STD_FRAC = 0.085

# Brand search: queries scale with demand and with what upper-funnel media
# create; brand clicks are bought at a cost per click.
_BRAND_SPEND_SHARE = 0.40
_BRAND_UPPER_FUNNEL = 0.6
_BRAND_CPC = 0.32
# Brand queries react more than sales to a market-wide demand shock (people
# search for the brand before they buy).
_BRAND_SHOCK_SENSITIVITY = 4.5

# Impressions per GBP of spend.
_IMPR_PER_SPEND = {"search": 40.0, "meta": 60.0, "youtube": 25.0, "ooh": 300.0}
_IMPR_NOISE_SIGMA = 0.10

_COLUMN_ORDER = [
    "region",
    "week",
    "date",
    "tv_grps",
    "tv_spend",
    "search_spend",
    "search_impr",
    "search_brand_spend",
    "search_generic_impr",
    "meta_spend",
    "meta_impr",
    "youtube_spend",
    "youtube_reach",
    "youtube_frequency",
    "ooh_spend",
    "ooh_impr",
    "price",
    "promo_depth",
    "distribution",
    "stock",
    "temperature",
    "rainfall",
    "launch",
    "category_search",
    "brand_search_volume",
    "revenue",
]


def _ar1(rng: np.random.Generator, shape: tuple[int, ...], sd: float) -> np.ndarray:
    """Stationary AR(1) series along the last axis with marginal sd ``sd``."""
    innov = rng.normal(0.0, sd * np.sqrt(1 - _SHOCK_PHI**2), shape)
    out = np.empty(shape)
    out[..., 0] = rng.normal(0.0, sd, shape[:-1])
    for t in range(1, shape[-1]):
        out[..., t] = _SHOCK_PHI * out[..., t - 1] + innov[..., t]
    return out


def _in(woy: np.ndarray, weeks: range) -> np.ndarray:
    return ((woy >= weeks.start) & (woy < weeks.stop)).astype(float)


def _shape(driver: np.ndarray, ch: str) -> tuple[np.ndarray, float]:
    """Adstocked, Hill-saturated shape of one region's driver, and its half point."""
    adstocked = geometric_adstock(driver, alpha=ADSTOCK_ALPHA[ch], l_max=12, normalize=True)
    half = HALF_MULT[ch] * float(np.mean(adstocked))
    return hill_saturation(adstocked, half=half, slope=HILL_SLOPE[ch]), half


def _build(n_weeks: int, n_regions: int, seed: int) -> tuple[np.random.Generator, list[dict], dict]:
    """Shared generation core used by every public entry point.

    Returns the RNG (already advanced, so a caller can keep drawing from the
    same stream), the per-region data, and the per-channel scale that makes
    each channel's aggregate ROAS equal ``GROUND_TRUTH`` exactly.
    """
    rng = np.random.default_rng(seed)
    T, R = n_weeks, n_regions
    week = np.arange(T)
    dates = _START_DATE + pd.to_timedelta(week * 7, unit="D")
    woy = week % 52
    race, xmas = _in(woy, _RACE_WEEKS), _in(woy, _XMAS_WEEKS)
    launch = np.zeros(T)
    launch_uplift = np.zeros(T)
    for start, uplift in LAUNCHES:
        on = (week >= start) & (week < start + LAUNCH_LENGTH)
        launch[on] = 1.0
        launch_uplift[on] = uplift
    seasonal = np.sin(2 * np.pi * woy / 52) + 0.5 * np.cos(2 * np.pi * woy / 52)

    size = rng.uniform(0.8, 1.2, R)  # region size
    years = T / 52.0
    budget_share = years * R / BOOK_N_REGIONS

    # ---- unobserved demand and what the data record of it
    shock_nat = _ar1(rng, (T,), _SHOCK_NAT_SD)
    shock_reg = _ar1(rng, (R, T), _SHOCK_REG_SD)
    shock = shock_nat[None, :] + shock_reg
    cat_season = 0.15 * np.cos(2 * np.pi * woy / 52) + 0.06 * np.sin(4 * np.pi * woy / 52)
    category = 100.0 * np.exp(
        cat_season[None, :]
        + _CATEGORY_SHOCK_LOADING * shock_nat[None, :]
        + 0.004 * week[None, :] / 52
        + rng.normal(0.0, 0.03, (R, T))
    )

    # ---- controls
    promo_on = rng.uniform(size=(R, T)) < (0.06 + 0.25 * race + 0.30 * xmas)[None, :]
    promo_depth = np.where(promo_on, rng.uniform(0.10, 0.30, (R, T)), 0.0)
    price = (
        PRICE_LIST
        * np.exp(rng.normal(0.0, _PRICE_NOISE, (R, T)))
        * (1 - _PROMO_PRICE_PASS_THROUGH * promo_depth)
    )
    distribution = np.clip(_DISTRIBUTION_MEAN + rng.normal(0, _DISTRIBUTION_STD, (R, T)), 0.5, 1.0)
    stock_drop = np.where(np.isin(week, list(_STOCKOUT_WEEKS)), _STOCKOUT_DEPTH, 0.0)
    stock = np.clip(
        _STOCK_MEAN
        + rng.normal(0, _STOCK_STD, (R, T))
        - stock_drop[None, :] * rng.uniform(0.5, 1.5, (R, 1)),
        0.6,
        1.0,
    )
    temperature = (
        10.5
        - 6.0 * np.cos(2 * np.pi * (woy - 2) / 52)[None, :]
        + rng.normal(0, 1.0, (R, 1))
        + rng.normal(0, 2.0, (R, T))
    )
    rainfall = rng.gamma(4.0, 1.0, (R, T)) * (
        3.8 + 1.2 * np.cos(2 * np.pi * woy / 52)[None, :]
    ) * rng.uniform(0.8, 1.25, (R, 1))

    # The new retail partner: true distribution steps up at the onboarding
    # week; the recorded series catches up ``lag_weeks`` later.
    ret = RETAILER_ONBOARDING
    ret_idx = int(ret["region"].split("_")[1])
    true_distribution = distribution.copy()
    if ret_idx < R and ret["week"] < T:
        true_distribution[ret_idx, ret["week"] :] += ret["step"]
        distribution[ret_idx, ret["week"] + ret["lag_weeks"] :] += ret["step"]
    true_distribution = np.clip(true_distribution, 0.5, 1.0)
    distribution = np.clip(distribution, 0.5, 1.0)

    # ---- media plans
    tv_flight = 0.25 + 2.6 * race + 2.2 * xmas
    tv_flight = tv_flight * np.where(rng.uniform(size=T) < 0.07, 2.0, 1.0)
    tv = size[:, None] * tv_flight[None, :] * rng.lognormal(0.0, 0.30, (R, T))

    social_plan = np.exp(_ar1(rng, (T,), 0.30))
    # The paid-social plan is held at its usual level around the Chapter 36
    # Meta holdout, so the test starts from business-as-usual spend.
    steady = [w for w in _STEADY_SOCIAL_WEEKS if w < T]
    if steady:
        social_plan[steady] = social_plan[: steady[0]].mean()
    meta = size[:, None] * social_plan * np.exp(
        _ar1(rng, (T,), 0.125)[None, :] + rng.normal(0, 0.11, (R, T))
    )
    youtube = size[:, None] * social_plan * np.exp(
        _ar1(rng, (T,), 0.125)[None, :] + rng.normal(0, 0.11, (R, T))
    )

    ooh = size[:, None] * rng.lognormal(0.0, 0.35, (R, T)) * np.where(
        rng.uniform(size=(R, T)) < 0.10, 3.0, 1.0
    )
    if ret_idx < R and ret["week"] < T:
        ooh[ret_idx, ret["week"] : ret["week"] + _RETAILER_OOH_WEEKS] *= _RETAILER_OOH_MULT

    generic = (
        size[:, None]
        * (category / 100.0) ** 2.1
        * np.exp(rng.normal(0.0, 0.03, T)[None, :] + rng.normal(0.0, 0.22, (R, T)))
    )

    def to_budget(x: np.ndarray, annual: float) -> np.ndarray:
        return x * annual * budget_share / x.sum()

    tv_grps = to_budget(tv, BOOK_ANNUAL_SPEND["tv"]) / TV_COST_PER_GRP
    meta = to_budget(meta, BOOK_ANNUAL_SPEND["meta"])
    youtube = to_budget(youtube, BOOK_ANNUAL_SPEND["youtube"])
    ooh = to_budget(ooh, BOOK_ANNUAL_SPEND["ooh"])
    # The past search-pause test: generic search switched off in a few
    # regions for eight weeks (brand search kept running).
    pause = SEARCH_PAUSE
    paused_regions = [r for r in pause["regions"] if r < R]
    generic[np.ix_(paused_regions, [w for w in pause["weeks"] if w < T])] = 0.0
    generic_spend = to_budget(generic, BOOK_ANNUAL_SPEND["search"] * (1 - _BRAND_SPEND_SHARE))
    generic_impr = generic_spend * _IMPR_PER_SPEND["search"] * rng.lognormal(
        0.0, _IMPR_NOISE_SIGMA, (R, T)
    )

    drivers = {"tv": tv_grps, "search": generic_impr, "meta": meta, "youtube": youtube, "ooh": ooh}
    shapes, halves = {}, {}
    for ch in CHANNELS:
        out = [_shape(drivers[ch][r], ch) for r in range(R)]
        shapes[ch] = np.array([o[0] for o in out])
        halves[ch] = np.array([o[1] for o in out])

    # ---- organic baseline
    level = (_BASE_LEVEL + _TREND_PER_WEEK * week + _BASE_LEVEL * _SEASONAL_AMPLITUDE * seasonal)
    demand = (
        np.exp(shock)
        * (1 + launch_uplift)[None, :]
        * (1 + _PROMO_UPLIFT_PER_DEPTH * promo_depth)
    )
    baseline = (
        level[None, :]
        * size[:, None]
        * true_distribution
        * (stock / _STOCK_MEAN)
        * np.power(PRICE_LIST / price, _PRICE_ELASTICITY)
        * np.exp(_TEMP_EFFECT * (temperature - 10.0) + _RAIN_EFFECT * (rainfall - 15.0))
        * np.exp(_CATEGORY_SEASON_EFFECT * cat_season)[None, :]
        * demand
    )

    # ---- brand search: demand that already exists, plus what upper-funnel media create
    upper = sum(shapes[ch] / shapes[ch].mean() for ch in ("tv", "meta", "youtube")) / 3.0
    brand_volume = (
        (baseline / baseline.mean())
        * np.exp(_BRAND_SHOCK_SENSITIVITY * shock)
        * (1 + _BRAND_UPPER_FUNNEL * (upper - 1))
        * rng.lognormal(0.0, 0.05, (R, T))
    )
    brand_spend = to_budget(brand_volume, BOOK_ANNUAL_SPEND["search"] * _BRAND_SPEND_SHARE)
    brand_volume = brand_spend / _BRAND_CPC * rng.lognormal(0.0, 0.04, (R, T))
    search_spend = generic_spend + brand_spend

    spend = {
        "tv": tv_grps * TV_COST_PER_GRP,
        "search": search_spend,
        "meta": meta,
        "youtube": youtube,
        "ooh": ooh,
    }
    scale = {ch: GROUND_TRUTH[ch] * spend[ch].sum() / shapes[ch].sum() for ch in CHANNELS}

    regions = []
    for r in range(R):
        regions.append(
            {
                "week": week,
                "date": dates,
                "baseline": baseline[r],
                "channel_spend": {ch: spend[ch][r] for ch in CHANNELS},
                "channel_driver": {ch: drivers[ch][r] for ch in CHANNELS},
                "channel_effect_shape": {ch: shapes[ch][r] for ch in CHANNELS},
                "channel_half": {ch: float(halves[ch][r]) for ch in CHANNELS},
                "columns": {
                    "tv_grps": tv_grps[r],
                    "tv_spend": tv_grps[r] * TV_COST_PER_GRP,
                    "search_spend": search_spend[r],
                    "search_brand_spend": brand_spend[r],
                    "search_generic_impr": generic_impr[r],
                    "meta_spend": meta[r],
                    "youtube_spend": youtube[r],
                    "ooh_spend": ooh[r],
                    "price": price[r],
                    "promo_depth": promo_depth[r],
                    "distribution": distribution[r],
                    "stock": stock[r],
                    "temperature": temperature[r],
                    "rainfall": rainfall[r],
                    "launch": launch,
                    "category_search": category[r],
                    "brand_search_volume": brand_volume[r],
                },
                "true_distribution": true_distribution[r],
            }
        )
    return rng, regions, scale


def ground_truth_contributions(
    n_weeks: int = 156, n_regions: int = BOOK_N_REGIONS, seed: int = RANDOM_SEED
) -> dict[str, np.ndarray]:
    """Return each channel's by-construction revenue contribution.

    For the same ``n_weeks``, ``n_regions`` and ``seed``, each returned array
    is row-for-row aligned with ``generate_sedge``'s DataFrame (region, then
    week). This is the supported way to recover the exact per-channel
    contribution behind ``GROUND_TRUTH``.
    """
    _, regions, scale = _build(n_weeks, n_regions, seed)
    return {
        ch: np.concatenate([region["channel_effect_shape"][ch] * scale[ch] for region in regions])
        for ch in CHANNELS
    }


def true_marginal_roas(
    n_weeks: int = 156,
    n_regions: int = BOOK_N_REGIONS,
    seed: int = RANDOM_SEED,
    bump: float = 0.01,
) -> dict[str, float]:
    """Return each channel's by-construction marginal ROAS at actual spend.

    Raises the channel's spend in every region-week by ``bump`` (1 per cent
    by default), pushes the matching rise in its driver through the same
    adstock and saturation (half-saturation point held fixed) and divides
    the extra revenue by the extra spend. The generator is tuned to the
    book's marginal ledger (see ``HALF_MULT``).
    """
    _, regions, scale = _build(n_weeks, n_regions, seed)
    out = {}
    for ch in CHANNELS:
        extra_revenue = 0.0
        extra_spend = 0.0
        for region in regions:
            adstocked = geometric_adstock(
                region["channel_driver"][ch] * (1.0 + bump),
                alpha=ADSTOCK_ALPHA[ch],
                l_max=12,
                normalize=True,
            )
            shape = hill_saturation(
                adstocked, half=region["channel_half"][ch], slope=HILL_SLOPE[ch]
            )
            extra_revenue += (shape - region["channel_effect_shape"][ch]).sum() * scale[ch]
            extra_spend += (region["channel_spend"][ch] * bump).sum()
        out[ch] = float(extra_revenue / extra_spend)
    return out


def correct_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Repair the late distribution update described in Chapter 35.

    The commercial team's onboarding date for the new retail partner
    (``RETAILER_ONBOARDING``) says the step in distribution happened
    ``lag_weeks`` before the series shows it. This copies the step back to
    the onboarding week in that region and returns a corrected copy.
    """
    ret = RETAILER_ONBOARDING
    out = df.copy()
    rows = (out["region"] == ret["region"]) & out["week"].between(
        ret["week"], ret["week"] + ret["lag_weeks"] - 1
    )
    out.loc[rows, "distribution"] += ret["step"]
    return out


def generate_sedge(
    n_weeks: int = 156, n_regions: int = BOOK_N_REGIONS, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Generate the synthetic Sedge weekly panel, as recorded.

    One row per region-week, with the columns Chapter 34 lists: Television
    GRPs and spend; Paid Search spend and impressions, split into brand
    spend and generic impressions; Meta spend and impressions; YouTube
    spend, reach and frequency; OOH spend and impressions; average selling
    price, promotion depth, distribution, stock availability, temperature,
    rainfall, the launch indicator, the category search index, branded
    search volume and net revenue. The ``distribution`` column is the
    recorded series, late update included (see ``correct_distribution``).
    """
    rng, regions, scale = _build(n_weeks, n_regions, seed)

    frames = []
    for i, region in enumerate(regions):
        n = n_weeks
        cols = region["columns"]
        data = {"region": f"region_{i}", "week": region["week"], "date": region["date"], **cols}
        contributions = sum(region["channel_effect_shape"][ch] * scale[ch] for ch in CHANNELS)

        def impressions(spend, ch, n=n):
            return spend * _IMPR_PER_SPEND[ch] * rng.lognormal(0.0, _IMPR_NOISE_SIGMA, n)

        brand_impr = region["columns"]["search_brand_spend"] / _BRAND_CPC * 6.0
        data["search_impr"] = cols["search_generic_impr"] + brand_impr * rng.lognormal(0.0, 0.05, n)
        data["meta_impr"] = impressions(cols["meta_spend"], "meta")
        youtube_impr = impressions(cols["youtube_spend"], "youtube")
        frequency = 1.6 + 0.9 * cols["youtube_spend"] / cols["youtube_spend"].mean()
        data["youtube_frequency"] = frequency * rng.lognormal(0.0, 0.05, n)
        data["youtube_reach"] = youtube_impr / data["youtube_frequency"]
        data["ooh_impr"] = impressions(cols["ooh_spend"], "ooh")

        noise = rng.normal(0.0, _NOISE_STD_FRAC * region["baseline"].mean(), n)
        data["revenue"] = region["baseline"] + contributions + noise
        frames.append(pd.DataFrame(data))

    df = pd.concat(frames, ignore_index=True)
    return df[_COLUMN_ORDER]
