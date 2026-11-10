import numpy as np

from mmm_book.adstock import geometric_adstock
from mmm_book.saturation import hill_saturation


def test_adstock_length_preserved():
    x = np.array([1.0, 0.0, 0.0, 0.0])
    out = geometric_adstock(x, alpha=0.5, l_max=4, normalize=False)
    assert out.shape == x.shape


def test_adstock_carryover_decays():
    x = np.array([1.0, 0.0, 0.0, 0.0])
    out = geometric_adstock(x, alpha=0.5, l_max=4, normalize=False)
    # impulse decays geometrically: 1, 0.5, 0.25, 0.125
    np.testing.assert_allclose(out, [1.0, 0.5, 0.25, 0.125], rtol=1e-6)


def test_adstock_normalize_sums_weights_to_one():
    x = np.ones(10)
    out = geometric_adstock(x, alpha=0.5, l_max=6, normalize=True)
    # steady-state normalised adstock of constant ones tends to 1
    assert abs(out[-1] - 1.0) < 1e-6


def test_hill_monotonic_and_bounded():
    x = np.linspace(0, 100, 50)
    out = hill_saturation(x, half=20.0, slope=1.0)
    assert np.all(np.diff(out) >= 0)
    assert out.min() >= 0.0
    assert out.max() < 1.0


def test_hill_half_point():
    out = hill_saturation(np.array([20.0]), half=20.0, slope=1.0)
    np.testing.assert_allclose(out, [0.5], rtol=1e-6)


def test_adstock_reproduces_book_chapter_13_table():
    # Chapter 13: 100,000 of TV in week 1 and nothing after, alpha = 0.5.
    x = np.array([100_000.0, 0.0, 0.0, 0.0])
    out = geometric_adstock(x, alpha=0.5, l_max=12, normalize=False)
    np.testing.assert_allclose(out, [100_000, 50_000, 25_000, 12_500])


def test_adstock_conventions_differ_only_by_constant():
    x = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
    raw = geometric_adstock(x, alpha=0.5, l_max=12, normalize=False)
    norm = geometric_adstock(x, alpha=0.5, l_max=12, normalize=True)
    expected_factor = 1.0 / (0.5 ** np.arange(12)).sum()
    np.testing.assert_allclose(norm / raw, expected_factor)
