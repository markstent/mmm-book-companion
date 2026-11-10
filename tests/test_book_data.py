"""The Sedge Shoes data facts Appendix F and Chapter 34 print.

Appendix F prints weeks 6 to 15 of ``region_0`` in three tables, rounded for
the page, and states the panel's shape and totals. These tests regenerate the
panel and check every printed cell, at the precision the book prints it,
so the tables cannot drift from ``generate_sedge``.
"""

import pandas as pd
import pytest

from mmm_book import RANDOM_SEED
from mmm_book.sedge import BOOK_ANNUAL_SPEND, CHANNELS, generate_sedge

WEEKS_SHOWN = range(6, 16)

# Appendix F, "Sedge rows: outcome and controls": column -> (decimals, values)
CONTROLS = {
    "revenue": (0, [59292, 56203, 67016, 62809, 73339, 62442, 97967, 85937, 73089, 59848]),
    "price": (2, [92.98, 96.14, 87.76, 91.52, 87.46, 93.95, 79.55, 82.12, 92.08, 96.31]),
    "promo_depth": (2, [0.12, 0.00, 0.00, 0.00, 0.18, 0.00, 0.30, 0.21, 0.12, 0.00]),
    "distribution": (
        3,
        [0.834, 0.894, 0.868, 0.884, 0.832, 0.852, 0.862, 0.833, 0.847, 0.832],
    ),
    "stock": (3, [0.972, 0.969, 0.976, 0.970, 0.970, 0.968, 0.981, 0.970, 0.986, 0.972]),
    "temperature": (1, [8.3, 4.2, 4.4, 7.3, 8.9, 6.7, 10.1, 12.2, 12.5, 11.6]),
    "rainfall": (1, [29.4, 31.4, 4.1, 7.3, 22.7, 15.6, 27.0, 5.4, 15.0, 17.1]),
    "category_search": (
        1,
        [121.1, 114.7, 117.1, 120.8, 109.0, 106.1, 106.9, 105.3, 96.5, 92.3],
    ),
}
DATES = [
    "2022-02-14",
    "2022-02-21",
    "2022-02-28",
    "2022-03-07",
    "2022-03-14",
    "2022-03-21",
    "2022-03-28",
    "2022-04-04",
    "2022-04-11",
    "2022-04-18",
]

# Appendix F, "Sedge rows: Television and Paid Search". Impressions are
# printed in thousands.
SEARCH = {
    "tv_grps": (2, [0.56, 1.02, 1.12, 1.01, 10.86, 6.34, 19.76, 10.33, 18.76, 6.22]),
    "tv_spend": (0, [225, 408, 449, 405, 4343, 2537, 7902, 4132, 7502, 2487]),
    "search_spend": (0, [2237, 1879, 2063, 1921, 2278, 1914, 3137, 2673, 2499, 1783]),
    "search_brand_spend": (0, [695, 513, 725, 776, 1188, 970, 2021, 1803, 1445, 1005]),
    "search_impr": (1, [83.8, 60.1, 66.0, 56.1, 57.1, 57.7, 90.3, 72.2, 70.5, 57.4]),
    "search_generic_impr": (1, [70.8, 51.1, 52.8, 42.3, 35.8, 41.4, 53.2, 37.6, 43.2, 37.5]),
    "brand_search_volume": (0, [2076, 1842, 2280, 2185, 4083, 2872, 6462, 5381, 4930, 2967]),
}

# Appendix F, "Sedge rows: Meta, YouTube and OOH". Impressions and reach are
# printed in thousands.
SOCIAL = {
    "meta_spend": (0, [1991, 1380, 2138, 2512, 2008, 2041, 1849, 2588, 1912, 2053]),
    "meta_impr": (1, [137.7, 71.9, 146.2, 142.9, 148.8, 127.9, 120.8, 152.8, 122.8, 131.9]),
    "youtube_spend": (0, [874, 712, 907, 1130, 817, 1135, 802, 1145, 862, 794]),
    "youtube_reach": (1, [8.5, 8.3, 9.0, 9.7, 8.4, 10.0, 10.6, 11.2, 8.4, 8.8]),
    "youtube_frequency": (2, [2.28, 2.27, 2.48, 2.70, 2.30, 2.63, 2.35, 2.83, 2.36, 2.19]),
    "ooh_spend": (0, [627, 665, 535, 1153, 1539, 480, 1166, 2228, 1261, 426]),
    "ooh_impr": (1, [223.9, 191.2, 182.6, 437.7, 442.9, 134.6, 281.8, 688.1, 392.9, 138.0]),
}
IN_THOUSANDS = {"search_impr", "search_generic_impr", "meta_impr", "youtube_reach", "ooh_impr"}


@pytest.fixture(scope="module")
def df():
    return generate_sedge(seed=RANDOM_SEED)


@pytest.fixture(scope="module")
def rows(df):
    sample = df[(df["region"] == "region_0") & df["week"].isin(WEEKS_SHOWN)]
    return sample.sort_values("week").reset_index(drop=True)


@pytest.mark.parametrize("table", [CONTROLS, SEARCH, SOCIAL], ids=["controls", "search", "social"])
def test_appendix_f_panel_rows(rows, table):
    assert rows["week"].tolist() == list(WEEKS_SHOWN)
    for col, (decimals, printed) in table.items():
        values = rows[col] / 1000 if col in IN_THOUSANDS else rows[col]
        assert [round(float(v), decimals) for v in values] == printed, col


def test_appendix_f_panel_dates_and_no_launch(rows):
    assert rows["date"].dt.strftime("%Y-%m-%d").tolist() == DATES
    assert (rows["launch"] == 0).all()


def test_appendix_f_panel_shape(df):
    assert df.shape == (2808, 26)
    assert not df.isna().any().any()
    assert df["region"].nunique() == 18
    assert sorted(df["region"].unique()) == sorted(f"region_{i}" for i in range(18))
    assert df["week"].nunique() == 156
    assert (df["week"].min(), df["week"].max()) == (0, 155)
    assert df["date"].min() == pd.Timestamp("2022-01-03")  # Monday 3 January 2022
    assert df["date"].max() == pd.Timestamp("2024-12-23")  # Monday 23 December 2024
    assert df["date"].dt.dayofweek.eq(0).all()
    assert (df["tv_spend"] == df["tv_grps"] * 400).all()


def test_appendix_f_revenue_totals(df):
    assert round(df["revenue"].sum() / 1e6, 1) == 143.8
    assert round(df["revenue"].sum() / 3 / 1e6, 1) == 47.9
    assert round(df["revenue"].mean(), -2) == 51_200


def _yearly(df):
    spend_cols = [f"{ch}_spend" for ch in CHANNELS]
    year = df["week"] // 52
    out = df.groupby(year)[["revenue", *spend_cols]].sum() / 1e6
    out["spend"] = out[spend_cols].sum(axis=1)
    return out


def test_yearly_revenue_and_spend(df):
    y = _yearly(df)
    assert y["revenue"].round(2).tolist() == [47.25, 46.04, 50.56]
    assert y["spend"].round(2).tolist() == [6.10, 5.44, 6.46]
    # Chapter 34: an average year of the history is about 48 million
    assert round(y["revenue"].mean(), 1) == 47.9


def test_average_annual_channel_spend(df):
    y = _yearly(df)
    printed = {"search": 1.40, "meta": 1.50, "youtube": 0.80, "tv": 1.40, "ooh": 0.90}
    for ch, value in printed.items():
        assert round(y[f"{ch}_spend"].mean(), 2) == value, ch
        assert BOOK_ANNUAL_SPEND[ch] == pytest.approx(value * 1e6)
    assert round(y["spend"].mean(), 2) == 6.00
