"""Shared matplotlib style for the companion notebooks."""

import matplotlib.pyplot as plt


def mmm_style() -> None:
    """Apply a clean, consistent plotting style."""
    plt.rcParams.update(
        {
            "figure.figsize": (9, 5),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 11,
        }
    )
