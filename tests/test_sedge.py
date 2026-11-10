import numpy as np

from mmm_book.sedge import (
    ADSTOCK_ALPHA,
    CHANNELS,
    GROUND_TRUTH,
    HILL_SLOPE,
    TV_COST_PER_GRP,
    generate_sedge,
    ground_truth_contributions,
)


def test_shape_and_columns():
    df = generate_sedge(n_weeks=104, n_regions=3)
    assert len(df) == 104 * 3
    for col in ["region", "week", "date", "revenue", "tv_grps", "search_spend"]:
        assert col in df.columns


def test_seed_stability():
    a = generate_sedge(seed=42)
    b = generate_sedge(seed=42)
    assert a["revenue"].equals(b["revenue"])


def test_revenue_scale_is_book_like():
    # book Sedge Shoes closes near 48 million pounds per year across regions
    df = generate_sedge(n_weeks=52)
    annual = df["revenue"].sum()
    assert 30e6 < annual < 70e6


def test_transform_hyperparams_public():
    assert isinstance(ADSTOCK_ALPHA, dict)
    assert isinstance(HILL_SLOPE, dict)
    assert set(ADSTOCK_ALPHA.keys()) == set(CHANNELS)
    assert set(HILL_SLOPE.keys()) == set(CHANNELS)


def test_ground_truth_covers_all_channels():
    for ch in CHANNELS:
        assert ch in GROUND_TRUTH
        assert GROUND_TRUTH[ch] > 0


def test_ground_truth_roas_holds():
    df = generate_sedge(n_weeks=104, n_regions=3)
    contrib = ground_truth_contributions(n_weeks=104, n_regions=3)

    for ch in CHANNELS:
        assert len(contrib[ch]) == len(df)

    for ch in ["search", "meta", "youtube", "ooh"]:
        implied_roas = contrib[ch].sum() / df[f"{ch}_spend"].sum()
        np.testing.assert_allclose(implied_roas, GROUND_TRUTH[ch], rtol=1e-6)

    implied_tv_roas = contrib["tv"].sum() / (df["tv_grps"].sum() * TV_COST_PER_GRP)
    np.testing.assert_allclose(implied_tv_roas, GROUND_TRUTH["tv"], rtol=1e-6)


def test_chapter_34_columns_are_present():
    df = generate_sedge(n_weeks=52, n_regions=2)
    for col in [
        "tv_spend",
        "search_brand_spend",
        "search_generic_impr",
        "youtube_reach",
        "youtube_frequency",
        "ooh_impr",
        "promo_depth",
        "stock",
        "temperature",
        "rainfall",
        "launch",
        "category_search",
        "brand_search_volume",
    ]:
        assert col in df.columns
    np.testing.assert_allclose(df["tv_spend"], df["tv_grps"] * TV_COST_PER_GRP)
    assert (df["search_brand_spend"] < df["search_spend"]).all()


def test_price_is_a_running_shoe_price():
    df = generate_sedge()
    assert 85 < df["price"].mean() < 100


def test_chapter_34_correlations():
    df = generate_sedge()
    nat = df.groupby("week").agg(
        meta=("meta_spend", "sum"),
        youtube=("youtube_spend", "sum"),
        search=("search_spend", "sum"),
        category=("category_search", "mean"),
    )
    # Meta and YouTube: 0.85 nationally and 0.76 across the panel (Chapter 34)
    assert round(np.corrcoef(nat["meta"], nat["youtube"])[0, 1], 2) == 0.85
    assert round(np.corrcoef(df["meta_spend"], df["youtube_spend"])[0, 1], 2) == 0.76
    # Paid Search and the category search index: 0.72
    assert round(np.corrcoef(nat["search"], nat["category"])[0, 1], 2) == 0.72


def test_launches_and_the_week_141_launch_is_larger():
    from mmm_book.sedge import LAUNCHES

    df = generate_sedge()
    weeks = df.loc[df["launch"] > 0, "week"].unique()
    assert len(weeks) == 18
    assert [start for start, _ in LAUNCHES] == [62, 114, 141]
    assert LAUNCHES[2][1] > 2 * LAUNCHES[0][1]


def test_distribution_lag_and_correction():
    from mmm_book.sedge import RETAILER_ONBOARDING, correct_distribution

    df = generate_sedge()
    ret = RETAILER_ONBOARDING
    fixed = correct_distribution(df)
    region = df["region"] == ret["region"]
    lag = df["week"].between(ret["week"], ret["week"] + ret["lag_weeks"] - 1)
    changed = (fixed["distribution"] - df["distribution"]).abs() > 1e-12
    assert (changed == (region & lag)).all()
    assert ret["lag_weeks"] == 12  # Chapter 35: twelve weeks of positive residuals


def test_search_pause_is_in_the_data():
    from mmm_book.sedge import SEARCH_PAUSE

    df = generate_sedge()
    paused = df["region"].isin([f"region_{r}" for r in SEARCH_PAUSE["regions"]]) & df[
        "week"
    ].isin(list(SEARCH_PAUSE["weeks"]))
    assert (df.loc[paused, "search_generic_impr"] == 0).all()
    assert (df.loc[paused, "search_brand_spend"] > 0).all()
