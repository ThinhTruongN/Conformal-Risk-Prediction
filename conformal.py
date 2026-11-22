"""
Split-conformal prediction intervals for regression.

Intervals are symmetric around the model's prediction:
    [prediction - q, prediction + q]
where q is a conformal quantile of absolute calibration residuals.

  - Standard split conformal: one q for everyone, the finite-sample-corrected
    (1 - alpha) empirical quantile.
  - Group-specific conformal: one q per (known) group.
  - Weighted conformal (Tibshirani et al., 2019): a per-test-point q from a
    weighted quantile, with weights given by the likelihood ratio
    w(x) = dP_test(x) / dP_calib(x). Valid under covariate shift.
  - Normalized scores: |y - mu(x)| / sigma(x), giving intervals
    mu(x) +/- q * sigma(x) whose width tracks estimated local difficulty.
  - Conformalized quantile regression (CQR; Romano et al., 2019): score
    max(lo(x) - y, y - hi(x)), giving [lo(x) - q, hi(x) + q].

Online methods (adaptive conformal inference) live in online.py.

When there are too few calibration points to reach the required quantile,
q is +inf (an infinite interval). That is the correct finite-sample answer,
not a numerical failure, so it is kept rather than clipped.
"""

from __future__ import annotations

from math import ceil

import numpy as np


def quantile_of_sorted(sorted_scores, alpha: float) -> float:
    """Finite-sample-corrected (1 - alpha) conformal quantile of already-sorted scores.

    Returns +inf when ceil((n + 1)(1 - alpha)) > n (too few scores to certify
    the level, which includes alpha <= 0) and 0 when alpha >= 1. Accepting any
    real alpha matters for adaptive conformal inference, whose alpha_t can leave [0, 1].
    """
    n = len(sorted_scores)
    rank = ceil((n + 1) * (1 - alpha))
    if rank > n:
        return float("inf")
    if rank < 1:
        return 0.0
    return float(sorted_scores[rank - 1])


def conformal_width(residuals: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1 - alpha) quantile of absolute residuals."""
    return quantile_of_sorted(np.sort(np.abs(residuals)), alpha)


def calibrate(model, X_calib: np.ndarray, y_calib: np.ndarray, alpha: float) -> float:
    """Standard (pooled) split-conformal calibration: one width for everyone."""
    residuals = y_calib - model.predict(X_calib)
    return conformal_width(residuals, alpha)


def calibrate_by_group(
    model, X_calib: np.ndarray, y_calib: np.ndarray, group_calib: np.ndarray, alpha: float
) -> dict[str, float]:
    """Group-specific split-conformal calibration: one width per group."""
    residuals = y_calib - model.predict(X_calib)
    return {g: conformal_width(residuals[group_calib == g], alpha) for g in np.unique(group_calib)}


def weighted_conformal_widths(
    calib_scores: np.ndarray,
    calib_log_weights: np.ndarray,
    test_log_weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Weighted split-conformal quantile, one per test point (Tibshirani et al., 2019).

    For a test point with weight w, the interval half-width is the (1 - alpha)
    quantile of the distribution

        sum_i p_i * delta(V_i) + p_test * delta(+inf),
        p_i = w_i / (sum_j w_j + w),   p_test = w / (sum_j w_j + w),

    where V_i are the absolute calibration residuals and w_i their weights.
    Weights are passed on the log scale to avoid overflow under strong shift;
    only their ratios matter, so a common constant is subtracted.
    """
    calib_scores = np.abs(np.asarray(calib_scores, dtype=float))
    calib_log_weights = np.asarray(calib_log_weights, dtype=float)
    test_log_weights = np.asarray(test_log_weights, dtype=float)

    shift = max(calib_log_weights.max(), test_log_weights.max())
    w_calib = np.exp(calib_log_weights - shift)
    w_test = np.exp(test_log_weights - shift)

    order = np.argsort(calib_scores)
    sorted_scores = calib_scores[order]
    cum_w = np.cumsum(w_calib[order])
    total = cum_w[-1]

    # Smallest V_(k) with cum_w[k] / (total + w_test) >= 1 - alpha.
    threshold = (1 - alpha) * (total + w_test)
    idx = np.searchsorted(cum_w, threshold * (1 - 1e-12), side="left")
    widths = np.full(len(w_test), np.inf)
    finite = idx < len(sorted_scores)
    widths[finite] = sorted_scores[idx[finite]]
    return widths


def normalized_scores(model, sigma_model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|y - mu(x)| / sigma(x): residuals in units of estimated local noise."""
    return np.abs(y - model.predict(X)) / predict_sigma(sigma_model, X)


def predict_sigma(sigma_model, X: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Estimated local residual scale, floored to keep normalized scores finite."""
    return np.maximum(sigma_model.predict(X), floor)


def cqr_scores(lo_model, hi_model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CQR conformity score: how far y falls outside [lo(x), hi(x)] (negative if inside)."""
    return np.maximum(lo_model.predict(X) - y, y - hi_model.predict(X))


def predict_intervals(model, X: np.ndarray, width) -> tuple[np.ndarray, np.ndarray]:
    pred = model.predict(X)
    return pred - width, pred + width


def interval_stats(pred: np.ndarray, y: np.ndarray, width) -> dict[str, float]:
    """Coverage and size summaries for symmetric intervals pred +/- width.

    `width` may be a scalar or a per-point array and may contain +inf.
    """
    width = np.broadcast_to(np.asarray(width, dtype=float), pred.shape)
    return bounds_stats(pred - width, pred + width, y)


def bounds_stats(lower: np.ndarray, upper: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Coverage and size summaries for intervals [lower, upper].

    Reports interval length (upper - lower):
      avg_width     mean length (inf if any interval is infinite)
      median_width  median length (inf once more than half are infinite)
      frac_infinite share of test points with an infinite interval
    """
    covered = (y >= lower) & (y <= upper)
    length = np.maximum(upper - lower, 0.0)
    return dict(
        coverage=float(np.mean(covered)),
        avg_width=float(np.mean(length)),
        median_width=float(np.median(length)),
        frac_infinite=float(np.mean(np.isinf(length))),
    )


def evaluate(model, X: np.ndarray, y: np.ndarray, width) -> tuple[float, float]:
    """Return (empirical coverage, average interval length) for a scalar or per-point width."""
    stats = interval_stats(model.predict(X), y, width)
    return stats["coverage"], stats["avg_width"]


def evaluate_by_group(
    model, X: np.ndarray, y: np.ndarray, group: np.ndarray, widths: dict[str, float]
) -> tuple[float, float]:
    """Return (empirical coverage, average interval length) using per-group widths."""
    width_vec = np.array([widths[g] for g in group])
    return evaluate(model, X, y, width_vec)
