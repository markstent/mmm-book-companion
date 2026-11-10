"""Contribution, ROAS and share helpers."""

import numpy as np


def roas(contribution: np.ndarray, spend: np.ndarray) -> float:
    """Return on ad spend: total contribution divided by total spend."""
    return float(np.sum(contribution) / np.sum(spend))


def contribution_share(contributions: dict[str, np.ndarray]) -> dict[str, float]:
    """Each channel's share of total contribution, summing to one."""
    totals = {k: float(np.sum(v)) for k, v in contributions.items()}
    grand = sum(totals.values())
    return {k: v / grand for k, v in totals.items()}
