"""Statistical helper utilities shared across evaluation stages."""

from __future__ import annotations

import math
import warnings
from typing import Sequence

import numpy as np


def paired_ttest_p_value(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Return a robust paired t-test p-value or ``None`` when unavailable.

    Degenerate paired samples are normalized explicitly to avoid noisy SciPy
    warnings and inconsistent NaN handling.
    """
    if len(a) < 2 or len(a) != len(b):
        return None

    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return None

    deltas = right - left
    if np.allclose(deltas, 0.0, atol=0.0, rtol=0.0):
        return 1.0
    if np.allclose(deltas, deltas[0], atol=0.0, rtol=0.0):
        return 0.0

    try:
        from scipy import stats
    except Exception:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, p_value = stats.ttest_rel(left, right)
    except Exception:
        return None

    if p_value is None:
        return None
    p_value = float(p_value)
    return p_value if math.isfinite(p_value) else None
