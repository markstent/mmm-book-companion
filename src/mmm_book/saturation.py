"""Hill saturation (diminishing returns) transform."""

import numpy as np


def hill_saturation(x: np.ndarray, half: float, slope: float = 1.0) -> np.ndarray:
    """Hill transform mapping spend to a saturating [0, 1) response.

    half is the half-saturation point (where response is 0.5).
    slope controls how sharply returns diminish.
    """
    x = np.asarray(x, dtype=float)
    xs = np.power(np.clip(x, 0, None), slope)
    return xs / (np.power(half, slope) + xs)
