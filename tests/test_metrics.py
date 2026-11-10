import numpy as np

from mmm_book.metrics import contribution_share, roas
from mmm_book.plotting import mmm_style


def test_roas():
    assert roas(np.array([10.0, 10.0]), np.array([4.0, 6.0])) == 2.0


def test_contribution_share_sums_to_one():
    shares = contribution_share({"a": np.array([2.0]), "b": np.array([2.0])})
    assert abs(sum(shares.values()) - 1.0) < 1e-9
    assert abs(shares["a"] - 0.5) < 1e-9


def test_style_runs():
    mmm_style()
