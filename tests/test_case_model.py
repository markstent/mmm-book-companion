from types import SimpleNamespace

import numpy as np
import pymc as pm
import pytest

from mmm_book import RANDOM_SEED
from mmm_book.case_model import (
    FIRST_PRIORS,
    PRIORS,
    build_case_model,
    marginal_roas_draws,
    prepare_case_data,
    roas_table,
)
from mmm_book.sedge import CHANNELS, correct_distribution, generate_sedge


@pytest.fixture(scope="module")
def small():
    return correct_distribution(generate_sedge(n_weeks=78, n_regions=3))


def test_prepare_case_data_shapes(small):
    data = prepare_case_data(small, "model2")
    assert data.y.shape == (3, 78)
    for ch in CHANNELS:
        assert data.drivers[ch].shape == (3, 78)
    assert data.controls.shape[:2] == (3, 78)
    # one launch (weeks 62 to 67) inside 78 weeks
    assert data.events.shape == (78, 1)
    assert data.events[:, 0].sum() == 6


def test_model2_swaps_search_spend_for_generic_impressions(small):
    m1 = prepare_case_data(small, "model1")
    m2 = prepare_case_data(small, "model2")
    assert not np.allclose(m1.drivers["search"], m2.drivers["search"])
    np.testing.assert_allclose(m1.spend["search"], m2.spend["search"])


def test_priors_follow_chapter_35():
    assert PRIORS.roas["tv"][0] == 2.2
    assert PRIORS.roas["meta"][0] == 1.8
    # the prior predictive check tightens OOH and shortens Search's carryover
    assert PRIORS.roas["ooh"][1] < FIRST_PRIORS.roas["ooh"][1]
    a, b = PRIORS.adstock["search"]
    a0, b0 = FIRST_PRIORS.adstock["search"]
    assert a / (a + b) < a0 / (a0 + b0)


def test_first_ooh_prior_allows_fantastical_contributions():
    from scipy import stats

    # Chapter 35: above 20m a year on 0.9m of spend in a non-trivial share of draws
    median, sd = FIRST_PRIORS.roas["ooh"]
    p_first = stats.lognorm(sd, scale=median).sf(20e6 / 0.9e6)
    median, sd = PRIORS.roas["ooh"]
    p_final = stats.lognorm(sd, scale=median).sf(20e6 / 0.9e6)
    assert p_first > 0.01
    assert p_final < 1e-4


def test_case_model_builds_samples_and_summarises(small):
    data = prepare_case_data(small, "model2")
    model = build_case_model(data, PRIORS, calibrate=False, store_mu=False)
    with model:
        idata = pm.sample(
            draws=30,
            tune=30,
            chains=1,
            cores=1,
            init="adapt_diag",
            random_seed=RANDOM_SEED,
            progressbar=False,
        )
    table = roas_table(idata, data)
    assert list(table.index) == CHANNELS
    assert (table["roas_mean"] > 0).all()
    marginal = marginal_roas_draws(idata, data, max_draws=10)
    assert set(marginal) == set(CHANNELS)


def test_calibrated_model_adds_the_holdout_term():
    df = correct_distribution(generate_sedge())
    data = prepare_case_data(df, "model2")
    model = build_case_model(data, PRIORS, calibrate=True, store_mu=False)
    assert "holdout_lift" in model.named_vars
    assert "holdout_implied_lift" in model.named_vars


def test_train_weeks_limits_the_likelihood(small):
    data = prepare_case_data(small, "model2", train_weeks=52)
    model = build_case_model(data, PRIORS, store_mu=False)
    assert pm.draw(model["y"], random_seed=RANDOM_SEED).shape == (3, 52)


def test_training_window_uses_no_test_week_information(small):
    """Chapter 21: scaling and reference levels come from the training weeks alone."""
    data = prepare_case_data(small, "model2", train_weeks=52)
    changed = small.copy()
    late = changed["week"] >= 52
    for col in ("price", "temperature", "meta_spend", "category_search"):
        changed.loc[late, col] *= 1.5
    other = prepare_case_data(changed, "model2", train_weeks=52)
    np.testing.assert_allclose(data.controls[:, :52], other.controls[:, :52])
    np.testing.assert_allclose(data.trend, other.trend)
    np.testing.assert_allclose(data.reference_activity("meta"), other.reference_activity("meta"))
    # standardised over the training weeks: mean zero there, not over the panel
    np.testing.assert_allclose(data.controls[:, :52].mean(axis=(0, 1)), 0.0, atol=1e-9)
    np.testing.assert_allclose(data.trend[:52].mean(), 0.0, atol=1e-9)


def test_posterior_curves_reproduce_average_roas(small):
    """Per-draw curves at current spend give back each draw's own ROAS."""
    from mmm_book.case_model import posterior_curves

    data = prepare_case_data(small, "model2")
    model = build_case_model(data, PRIORS, calibrate=False, store_mu=False)
    with model:
        prior = pm.sample_prior_predictive(draws=20, random_seed=RANDOM_SEED)
    idata = SimpleNamespace(posterior=prior["prior"].to_dataset())
    curves = posterior_curves(idata, data)
    for ch in CHANNELS:
        roas = idata.posterior[f"roas_{ch}"].to_numpy().ravel()
        weekly = curves.draws(ch, curves.current_money[ch])
        np.testing.assert_allclose(weekly / curves.current_money[ch], roas, rtol=1e-6)
