"""Geometric adstock (carryover) transform."""

import numpy as np


def geometric_adstock(
    x: np.ndarray,
    alpha: float,
    l_max: int = 12,
    normalize: bool = True,
) -> np.ndarray:
    """Apply geometric adstock to a 1D media series.

    alpha is the retention rate in [0, 1). l_max is the maximum lag.

    Two conventions are supported:

    * ``normalize=False`` is the textbook form of Chapter 13,
      ``A_t = x_t + alpha * A_(t-1)``, truncated at ``l_max`` lags. A
      one-off 100 of spend with alpha = 0.5 gives 100, 50, 25, 12.5, ...,
      which is the worked table printed in the book.
    * ``normalize=True`` (the default here, used by the labs and the
      Sedge Shoes generator) rescales the lag weights to sum to one, so
      adstock does not inflate the total media level and a steady spend
      level maps to itself. The same 100 then gives about 51.6, 25.8,
      12.9, ... for ``l_max=12`` and alpha = 0.5.

    The two differ only by a constant factor, ``sum(alpha ** k)`` over the
    lags, so the temporal weighting is identical. pymc-marketing's
    ``GeometricAdstock`` component (the one the ``MMM`` class uses) also
    normalises by default, although its low-level ``geometric_adstock``
    function defaults to ``normalize=False``; check the version you use.
    """
    x = np.asarray(x, dtype=float)
    weights = alpha ** np.arange(l_max)
    if normalize:
        weights = weights / weights.sum()
    out = np.zeros_like(x)
    for lag, w in enumerate(weights):
        if lag == 0:
            out += w * x
        else:
            out[lag:] += w * x[:-lag]
    return out
