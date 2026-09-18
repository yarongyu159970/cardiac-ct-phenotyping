"""Graf IPCW Brier score and integrated Brier score.

Failures are weighted by G(T-) and observations still at risk after t by G(t), where G is the
reverse Kaplan-Meier censoring distribution estimated on the training sample. At tied times,
outcome events are processed before censorings.
"""

from __future__ import annotations

import numpy as np


def censor_km(durations, events) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=bool)
    if t.ndim != 1 or e.shape != t.shape or not len(t) or not np.isfinite(t).all() or (t < 0).any():
        raise ValueError("Censoring KM requires complete nonnegative times and matching events")
    unique, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    failures = np.bincount(inverse, weights=e.astype(int), minlength=len(unique))
    censored = counts - failures
    at_risk = len(t) - np.r_[0, np.cumsum(counts[:-1])]
    denominator = at_risk - failures
    if np.any((denominator <= 0) & (censored > 0)):
        raise ValueError("Invalid censoring risk set")
    hazard = np.zeros(len(unique))
    np.divide(censored, denominator, out=hazard, where=denominator > 0)
    return unique, np.cumprod(1 - hazard)


def graf_curve(train_times, train_events, eval_times, eval_events, probability, grid) -> np.ndarray:
    """Brier score at each grid time for predicted survival probabilities of shape (n, len(grid))."""
    t = np.asarray(eval_times, dtype=float)
    e = np.asarray(eval_events, dtype=bool)
    grid = np.asarray(grid, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if e.shape != t.shape or probability.shape != (len(t), len(grid)) or len(t) == 0:
        raise ValueError("Unexpected evaluation array shape")
    if not np.isfinite(t).all() or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("Finite times and a strictly increasing grid are required")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Survival probabilities must be finite and in [0, 1]")
    unique, g = censor_km(train_times, train_events)
    g_event = np.r_[1.0, g][np.searchsorted(unique, t, side="left")]
    g_grid = np.r_[1.0, g][np.searchsorted(unique, grid, side="right")]
    failed = e[:, None] & (t[:, None] <= grid[None, :])
    observed = t[:, None] > grid[None, :]
    if np.any((g_event <= 0) & failed.any(axis=1)) or np.any((g_grid <= 0) & observed.any(axis=0)):
        raise ValueError("Zero censoring survival in a required IPCW contribution")
    w_failed = np.divide(failed, g_event[:, None], out=np.zeros_like(probability), where=g_event[:, None] > 0)
    w_observed = np.divide(observed, g_grid[None, :], out=np.zeros_like(probability), where=g_grid[None, :] > 0)
    return (w_failed * probability**2 + w_observed * (1 - probability) ** 2).mean(axis=0)


def integrate_curve(scores, grid) -> float:
    grid = np.asarray(grid, dtype=float)
    if len(grid) < 2 or grid[-1] <= grid[0]:
        raise ValueError("IBS requires a positive integration interval")
    trapezoid = np.trapz if hasattr(np, "trapz") else np.trapezoid
    return float(trapezoid(scores, grid) / (grid[-1] - grid[0]))
