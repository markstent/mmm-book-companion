import numpy as np
import pymc as pm
import pytest

from mmm_book import RANDOM_SEED
from mmm_book.reference import (
    BETA_PRIOR_SHARE,
    build_controls,
    build_events,
    build_features,
    build_seasonality,
    build_treatments,
    national_aggregate,
    reference_model,
)
from mmm_book.sedge import CHANNELS, generate_sedge


def test_build_features_shape_and_range():
    df = generate_sedge(n_weeks=52, n_regions=2)
    nat = national_aggregate(df)
    feat = build_features(nat)

    assert set(feat.keys()) == set(CHANNELS)
    for ch in CHANNELS:
        arr = feat[ch]
        assert len(arr) == len(nat)
        assert np.all(np.isfinite(arr))
        assert arr.min() >= 0.0
        # features are scaled to average one, so a beta is an average weekly contribution
        assert arr.mean() == pytest.approx(1.0)


def test_reference_model_builds_and_samples():
    df = generate_sedge(n_weeks=52, n_regions=2)
    nat = national_aggregate(df)
    model = reference_model(nat)

    assert isinstance(model, pm.Model)

    with model:
        idata = pm.sample(
            draws=50,
            tune=50,
            chains=1,
            cores=1,
            progressbar=False,
            random_seed=RANDOM_SEED,
        )

    beta_names = {f"beta_{ch}" for ch in CHANNELS}
    assert beta_names.issubset(set(idata.posterior.data_vars))


def test_national_aggregate_averages_the_controls():
    df = generate_sedge(n_weeks=10, n_regions=3)
    nat = national_aggregate(df)
    assert nat["price"].between(df["price"].min(), df["price"].max()).all()
    assert nat["promo_depth"].between(0.0, 0.3).all()
    assert nat["stock"].between(df["stock"].min(), df["stock"].max()).all()
    assert nat["distribution"].between(0.5, 1.0).all()
    assert nat["revenue"].iloc[0] == pytest.approx(df.loc[df["week"] == 0, "revenue"].sum())


def test_build_controls_are_standardised():
    nat = national_aggregate(generate_sedge(n_weeks=104, n_regions=3))
    controls = build_controls(nat)
    # Chapter 15: seasonality and launches are baseline, promotion a non-media treatment.
    assert set(controls) == {
        "log_price",
        "distribution",
        "stock",
        "temperature",
        "rainfall",
        "category",
    }
    for name in ("log_price", "distribution", "stock", "category"):
        assert controls[name].mean() == pytest.approx(0.0, abs=1e-9)
        assert controls[name].std() == pytest.approx(1.0)
    promo = build_treatments(nat)["promo"]
    assert promo.mean() == pytest.approx(0.0, abs=1e-9)
    assert promo.std() == pytest.approx(1.0)


def test_baseline_terms():
    nat = national_aggregate(generate_sedge(n_weeks=104, n_regions=3))
    season = build_seasonality(nat)
    assert set(season) == {"sin", "cos"}
    assert np.allclose(season["sin"] ** 2 + season["cos"] ** 2, 1.0)
    launch = build_events(nat)["launch"]
    assert set(np.unique(launch)) <= {0.0, 1.0}


def test_reference_model_terms_and_priors():
    nat = national_aggregate(generate_sedge(n_weeks=104, n_regions=3))
    names = {rv.name for rv in reference_model(nat).free_RVs}
    assert {"intercept", "trend", "season_a", "season_b", "omega_launch", "delta_promo"} <= names
    assert {f"beta_{ch}" for ch in CHANNELS} <= names
    assert "gamma_promo" not in names

    baseline = {rv.name for rv in reference_model(nat, include_media=False).free_RVs}
    assert not any(name.startswith("beta_") for name in baseline)

    # Lab 17's choice: each beta's prior scale is a tenth of mean weekly revenue.
    y = nat["revenue"].to_numpy() / 1e6
    with reference_model(nat):
        draws = pm.draw(pm.modelcontext(None)["beta_tv"], draws=4000, random_seed=RANDOM_SEED)
    half_normal_sd = BETA_PRIOR_SHARE * y.mean() * np.sqrt(1 - 2 / np.pi)
    assert np.std(draws) == pytest.approx(half_normal_sd, rel=0.05)


def test_reference_model_held_out_split():
    nat = national_aggregate(generate_sedge(n_weeks=104, n_regions=3))
    model = reference_model(nat, n_train=80)
    assert pm.draw(model["y"], random_seed=RANDOM_SEED).shape == (80,)
    assert pm.draw(model["mu"], random_seed=RANDOM_SEED).shape == (104,)
    with pytest.raises(ValueError):
        reference_model(nat, n_train=0)
