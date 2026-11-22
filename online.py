"""
Online conformal prediction for data that arrive one at a time, with the true
label revealed after each prediction.

Adaptive conformal inference (ACI; Gibbs & Candes, 2021) treats the
miscoverage level as a control variable. After each step it updates

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t),   err_t = 1{y_t not covered},

so a miss lowers alpha_t (wider intervals) and a hit raises it (narrower).
Whatever the data do, the long-run miscoverage rate satisfies
|mean(err) - alpha| <= (max(alpha_1, 1 - alpha_1) + gamma) / (gamma * T), so ACI
reaches the target over time without any exchangeability assumption. alpha_t
can leave [0, 1]: alpha_t <= 0 gives an infinite interval, alpha_t >= 1 an
empty one.

Four online methods are compared, all producing symmetric intervals
pred_t +/- q_t:
  standard      fixed scores from the initial calibration set, fixed alpha
  aci           fixed initial scores, alpha_t updated by ACI
  rolling       scores from the most recent `window` labeled points, fixed alpha
  rolling_aci   rolling-window scores with alpha_t updated by ACI
"""

from __future__ import annotations

from bisect import bisect_left, insort

import numpy as np

from conformal import quantile_of_sorted

METHODS = ("standard", "aci", "rolling", "rolling_aci")


def run_online(
    calib_scores: np.ndarray,
    stream_scores: np.ndarray,
    alpha: float,
    gamma: float = 0.005,
    window: int = 500,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run every online method over one stream.

    calib_scores   absolute residuals from the initial (pre-stream) calibration set
    stream_scores  absolute residuals |y_t - pred_t| of the stream, in time order;
                   score t is only used *after* step t's interval is formed
    Returns {method: (covered[t], half_width[t])}.
    """
    T = len(stream_scores)
    fixed_sorted = np.sort(np.abs(calib_scores))

    # Rolling window starts as the most recent `window` calibration scores.
    recent = list(np.abs(calib_scores)[-window:])
    window_sorted = sorted(recent)
    oldest = 0  # index into `recent` of the next score to evict

    alpha_aci = alpha
    alpha_rolling_aci = alpha
    q_standard = quantile_of_sorted(fixed_sorted, alpha)

    covered = {m: np.zeros(T, dtype=bool) for m in METHODS}
    width = {m: np.zeros(T) for m in METHODS}

    for t in range(T):
        s = stream_scores[t]
        q = {
            "standard": q_standard,
            "aci": quantile_of_sorted(fixed_sorted, alpha_aci),
            "rolling": quantile_of_sorted(window_sorted, alpha),
            "rolling_aci": quantile_of_sorted(window_sorted, alpha_rolling_aci),
        }
        for m in METHODS:
            width[m][t] = q[m]
            covered[m][t] = s <= q[m]

        # Label revealed: update ACI levels, then slide the window.
        alpha_aci += gamma * (alpha - (not covered["aci"][t]))
        alpha_rolling_aci += gamma * (alpha - (not covered["rolling_aci"][t]))

        evict = recent[oldest]
        del window_sorted[bisect_left(window_sorted, evict)]
        recent[oldest] = s
        oldest = (oldest + 1) % len(recent)
        insort(window_sorted, s)

    return {m: (covered[m], width[m]) for m in METHODS}
