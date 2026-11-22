"""
All experiments: baseline, distribution-shift sweeps, and attempted fixes.

Every shift experiment is repeated over `n_trials` independent trials. In each
trial the calibration set and every test set are freshly drawn, so the reported
coverage reflects both sources of randomness in split conformal prediction.
The base model is trained once and held fixed, as in a deployed system.

Within a trial, all methods being compared are scored on the *same* test set
(a paired design), so differences between methods are not test-draw noise.

Every function returns a pandas DataFrame in long format with one row per
(shift level, method), plus these columns:
  coverage_mean  mean empirical coverage over trials
  coverage_std   std of empirical coverage over trials (trial-to-trial spread)
  coverage_se    standard error of coverage_mean (= coverage_std / sqrt(n_trials))
  width_mean     mean interval length (inf if any interval was infinite)
  width_median   median interval length over test points, then over trials
  frac_infinite  mean share of test points that got an infinite interval
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from conformal import (
    bounds_stats,
    calibrate,
    calibrate_by_group,
    conformal_width,
    cqr_scores,
    interval_stats,
    normalized_scores,
    predict_sigma,
    quantile_of_sorted,
    weighted_conformal_widths,
)
from data_generation import (
    N_FEATURES,
    covariate_shift_log_weight,
    make_data,
    make_stream,
    make_subgroup_data,
)
from online import METHODS as ONLINE_METHODS
from online import run_online

N_TRIALS = 100

# Seed-stream ids. Each draw is seeded with a tuple (stream, ..., trial), so
# every draw in every experiment is reproducible and independent of the others.
_SOURCE_CALIB = 1
_COV_TEST, _COV_RECAL, _COV_UNLAB_SRC, _COV_UNLAB_TGT = 10, 11, 12, 13
_NOISE_TEST = 20
_SUB_CALIB, _SUB_TEST = 30, 31
_SIZE_TEST, _SIZE_CALIB = 40, 41
_MODEL_TRAIN, _MODEL_TEST = 50, 51
_HET_TRAIN, _HET_CALIB, _HET_TEST = 60, 61, 62
_STREAM = 70


def make_default_model(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)


def _summarize(records: list[dict], keys: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    out = (
        df.groupby(keys, sort=False)
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            n_trials=("coverage", "size"),
            width_mean=("avg_width", "mean"),
            width_median=("median_width", "median"),
            frac_infinite=("frac_infinite", "mean"),
        )
        .reset_index()
    )
    out.insert(out.columns.get_loc("coverage_std") + 1, "coverage_se",
               out["coverage_std"] / np.sqrt(out["n_trials"]))
    return out.drop(columns="n_trials")


def _source_calibration(model, alpha: float, n_calib: int, trial: int):
    """Draw a calibration set from the unshifted (source) distribution.

    Returns (X_calib, absolute residuals, standard conformal width)."""
    X_calib, y_calib = make_data(n_calib, seed=(_SOURCE_CALIB, trial))
    scores = np.abs(y_calib - model.predict(X_calib))
    return X_calib, scores, conformal_width(scores, alpha)


def run_baseline(
    alpha: float = 0.10, n_train: int = 1500, n_calib: int = 500, n_test: int = 1000, seed: int = 0
) -> dict:
    """Fit the base model, calibrate at the nominal (1 - alpha) level, and
    evaluate on one unshifted test set (a single illustrative draw; the
    repeated-trial version of this number is the shift = 0 row of the
    covariate-shift experiment)."""
    X_train, y_train = make_data(n_train, seed=seed)
    X_calib, y_calib = make_data(n_calib, seed=seed + 1)
    X_test, y_test = make_data(n_test, seed=seed + 2)

    model = make_default_model(seed).fit(X_train, y_train)
    width = calibrate(model, X_calib, y_calib, alpha)
    stats = interval_stats(model.predict(X_test), y_test, width)

    return dict(
        model=model, width=width, alpha=alpha,
        coverage=stats["coverage"], avg_width=stats["avg_width"],
        X_train=X_train, y_train=y_train, X_calib=X_calib, y_calib=y_calib,
        X_test=X_test, y_test=y_test,
    )


def _estimated_log_weights(X_source: np.ndarray, X_target: np.ndarray):
    """Estimate log[p_target(x) / p_source(x)] with a probabilistic classifier.

    Fits logistic regression to tell unlabeled source inputs (label 0) from
    unlabeled target inputs (label 1); the odds, corrected for the class
    ratio, estimate the likelihood ratio. Logistic regression is well-specified
    here because the true log-ratio between two equal-covariance Gaussians is
    linear in x. Returns a function mapping X -> estimated log weights.
    """
    X = np.vstack([X_source, X_target])
    labels = np.concatenate([np.zeros(len(X_source)), np.ones(len(X_target))])
    clf = LogisticRegression(max_iter=1000).fit(X, labels)
    log_prior_ratio = np.log(len(X_source) / len(X_target))
    return lambda X_new: clf.decision_function(X_new) + log_prior_ratio


def covariate_shift_experiment(
    model, alpha: float, shifts: list[float], n_trials: int = N_TRIALS,
    n_calib: int = 500, n_test: int = 1000, n_unlabeled: int = 1000, noise_std: float = 1.0,
) -> pd.DataFrame:
    """Covariate shift: test inputs ~ N(shift, 1). Compares, on shared test sets:

      standard            calibrated once on unshifted data (no correction)
      recalibrated        recalibrated on a fresh *labeled* sample from the
                          shifted distribution (idealized: needs new labels)
      weighted_oracle     weighted conformal with the exact likelihood ratio
      weighted_estimated  weighted conformal with a likelihood ratio estimated
                          from *unlabeled* source and target inputs only
    """
    records = []
    for r in range(n_trials):
        X_calib, scores, width_std = _source_calibration(model, alpha, n_calib, r)
        for i, shift in enumerate(shifts):
            X_test, y_test = make_data(n_test, seed=(_COV_TEST, i, r), x_loc=shift, noise_std=noise_std)
            pred = model.predict(X_test)

            X_new, y_new = make_data(n_calib, seed=(_COV_RECAL, i, r), x_loc=shift, noise_std=noise_std)
            width_recal = calibrate(model, X_new, y_new, alpha)

            width_oracle = weighted_conformal_widths(
                scores,
                covariate_shift_log_weight(X_calib, x_loc=shift),
                covariate_shift_log_weight(X_test, x_loc=shift),
                alpha,
            )

            X_src, _ = make_data(n_unlabeled, seed=(_COV_UNLAB_SRC, i, r))
            X_tgt, _ = make_data(n_unlabeled, seed=(_COV_UNLAB_TGT, i, r), x_loc=shift)
            log_w = _estimated_log_weights(X_src, X_tgt)
            width_est = weighted_conformal_widths(scores, log_w(X_calib), log_w(X_test), alpha)

            for method, width in [
                ("standard", width_std),
                ("recalibrated", width_recal),
                ("weighted_oracle", width_oracle),
                ("weighted_estimated", width_est),
            ]:
                records.append(dict(shift=shift, method=method, **interval_stats(pred, y_test, width)))
    return _summarize(records, ["shift", "method"])


def noise_shift_experiment(
    model, alpha: float, noise_levels: list[float], n_trials: int = N_TRIALS,
    n_calib: int = 500, n_test: int = 1000,
) -> pd.DataFrame:
    """Noise shift: calibrate on noise_std = 1, test at higher noise levels."""
    records = []
    for r in range(n_trials):
        _, _, width = _source_calibration(model, alpha, n_calib, r)
        for i, noise_std in enumerate(noise_levels):
            X_test, y_test = make_data(n_test, seed=(_NOISE_TEST, i, r), noise_std=noise_std)
            stats = interval_stats(model.predict(X_test), y_test, width)
            records.append(dict(noise_std=noise_std, method="standard", **stats))
    return _summarize(records, ["noise_std", "method"])


def subgroup_shift_experiment(
    model, alpha: float, frac_bs: list[float], baseline_frac_b: float = 0.2,
    n_trials: int = N_TRIALS, n_calib: int = 500, n_test: int = 1000,
    noise_std_a: float = 1.0, noise_std_b: float = 2.5,
) -> pd.DataFrame:
    """Subgroup (mixture) shift.

    Calibration data is drawn from the baseline mixture (baseline_frac_b of the
    sample from high-noise group B), so at frac_b == baseline_frac_b there is no
    shift and standard conformal should hit its target. Test sets then shift the
    mixture weight. Compares a pooled width ("standard") against per-group
    widths ("group_specific") on shared test sets.
    """
    records = []
    for r in range(n_trials):
        X_calib, y_calib, g_calib = make_subgroup_data(
            n_calib, seed=(_SUB_CALIB, r), frac_b=baseline_frac_b,
            noise_std_a=noise_std_a, noise_std_b=noise_std_b,
        )
        pooled_width = calibrate(model, X_calib, y_calib, alpha)
        group_widths = calibrate_by_group(model, X_calib, y_calib, g_calib, alpha)

        for i, frac_b in enumerate(frac_bs):
            X_test, y_test, g_test = make_subgroup_data(
                n_test, seed=(_SUB_TEST, i, r), frac_b=frac_b,
                noise_std_a=noise_std_a, noise_std_b=noise_std_b,
            )
            pred = model.predict(X_test)
            group_vec = np.array([group_widths[g] for g in g_test])
            records.append(dict(frac_b=frac_b, method="standard", **interval_stats(pred, y_test, pooled_width)))
            records.append(dict(frac_b=frac_b, method="group_specific", **interval_stats(pred, y_test, group_vec)))
    return _summarize(records, ["frac_b", "method"])


def calibration_size_experiment(
    model, alpha: float, calib_sizes: list[int], shift: float, n_trials: int = N_TRIALS,
    n_test: int = 1000, noise_std: float = 1.0,
) -> pd.DataFrame:
    """At a fixed covariate shift, recalibrate on labeled shifted-distribution
    calibration sets of varying size. Each trial draws a fresh test set (shared
    across sizes) and a fresh calibration set per size, so the spread reflects
    the full randomness of a single deployment."""
    records = []
    for r in range(n_trials):
        X_test, y_test = make_data(n_test, seed=(_SIZE_TEST, r), x_loc=shift, noise_std=noise_std)
        pred = model.predict(X_test)
        for size in calib_sizes:
            X_calib, y_calib = make_data(size, seed=(_SIZE_CALIB, size, r), x_loc=shift, noise_std=noise_std)
            width = calibrate(model, X_calib, y_calib, alpha)
            records.append(dict(calib_size=size, method="recalibrated", **interval_stats(pred, y_test, width)))
    df = pd.DataFrame(records)
    out = _summarize(records, ["calib_size", "method"])
    out.insert(out.columns.get_loc("width_mean") + 1, "width_std",
               df.groupby("calib_size", sort=False)["avg_width"].std().to_numpy())
    return out


def model_comparison_experiment(
    alpha: float, noise_levels: list[float], n_trials: int = N_TRIALS,
    n_train: int = 1500, n_calib: int = 500, n_test: int = 1000, seed: int = 700,
) -> pd.DataFrame:
    """Train several regression models on the same data, conformalize each on
    the same calibration draws, and compare coverage/width under noise shift
    on the same test draws."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=seed),
    }
    X_train, y_train = make_data(n_train, seed=(_MODEL_TRAIN, seed))
    for model in models.values():
        model.fit(X_train, y_train)

    records = []
    for r in range(n_trials):
        X_calib, y_calib = make_data(n_calib, seed=(_SOURCE_CALIB, r))
        tests = [make_data(n_test, seed=(_MODEL_TEST, i, r), noise_std=s) for i, s in enumerate(noise_levels)]
        for name, model in models.items():
            width = calibrate(model, X_calib, y_calib, alpha)
            for noise_std, (X_test, y_test) in zip(noise_levels, tests):
                stats = interval_stats(model.predict(X_test), y_test, width)
                records.append(dict(model=name, noise_std=noise_std, **stats))
    return _summarize(records, ["model", "noise_std"])


# ---------------------------------------------------------------------------
# Locally adaptive intervals (normalized scores, CQR) on heteroscedastic data
# ---------------------------------------------------------------------------

X3_BIN_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, np.inf]
X3_BIN_LABELS = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2+"]


def fit_adaptive_models(alpha: float, n_train: int = 1500, seed: int = 0) -> dict:
    """Fit everything the adaptive methods need on one heteroscedastic training set.

      mean   random forest for E[y | x] (used by standard and normalized)
      sigma  random forest for the residual scale, fit to |out-of-bag residuals|
             of `mean` so it sees honest (not overfit) residuals
      lo/hi  gradient-boosted quantile regressors at alpha/2 and 1 - alpha/2 (CQR)
    """
    X, y = make_data(n_train, seed=(_HET_TRAIN, seed), hetero=True)
    mean = RandomForestRegressor(n_estimators=300, oob_score=True, random_state=seed, n_jobs=-1).fit(X, y)
    oob_abs_resid = np.abs(y - mean.oob_prediction_)
    sigma = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=20, random_state=seed, n_jobs=-1
    ).fit(X, oob_abs_resid)
    lo = HistGradientBoostingRegressor(loss="quantile", quantile=alpha / 2, random_state=seed).fit(X, y)
    hi = HistGradientBoostingRegressor(loss="quantile", quantile=1 - alpha / 2, random_state=seed).fit(X, y)
    return dict(mean=mean, sigma=sigma, lo=lo, hi=hi)


def _adaptive_bounds(models: dict, q: dict, X: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    pred = models["mean"].predict(X)
    sig = predict_sigma(models["sigma"], X)
    lo, hi = models["lo"].predict(X), models["hi"].predict(X)
    return {
        "standard": (pred - q["standard"], pred + q["standard"]),
        "normalized": (pred - q["normalized"] * sig, pred + q["normalized"] * sig),
        "cqr": (lo - q["cqr"], hi + q["cqr"]),
    }


def adaptive_width_experiment(
    alpha: float, x3_shifts: list[float], n_trials: int = N_TRIALS,
    n_calib: int = 500, n_test: int = 1000, seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Heteroscedastic data (noise std 0.2 + |x3|) under a shift of the mean of x3,
    which moves test points toward the noisy region while keeping the other
    features in place. Compares on shared test sets:

      standard    fixed-width split conformal (|residual| score)
      normalized  |residual| / sigma(x) score, interval mu(x) +/- q sigma(x)
      cqr         conformalized quantile regression

    Returns (marginal results by shift, conditional coverage by |x3| bin at no shift).
    """
    models = fit_adaptive_models(alpha, seed=seed)
    records, cond_records = [], []
    for r in range(n_trials):
        X_cal, y_cal = make_data(n_calib, seed=(_HET_CALIB, r), hetero=True)
        q = {
            "standard": conformal_width(y_cal - models["mean"].predict(X_cal), alpha),
            "normalized": quantile_of_sorted(np.sort(normalized_scores(models["mean"], models["sigma"], X_cal, y_cal)), alpha),
            "cqr": quantile_of_sorted(np.sort(cqr_scores(models["lo"], models["hi"], X_cal, y_cal)), alpha),
        }
        for i, shift in enumerate(x3_shifts):
            loc = np.zeros(N_FEATURES)
            loc[3] = shift
            X_test, y_test = make_data(n_test, seed=(_HET_TEST, i, r), x_loc=loc, hetero=True)
            bounds = _adaptive_bounds(models, q, X_test)
            for method, (lo, hi) in bounds.items():
                records.append(dict(x3_shift=shift, method=method, **bounds_stats(lo, hi, y_test)))

            if shift == 0.0:
                bins = np.digitize(np.abs(X_test[:, 3]), X3_BIN_EDGES[1:-1])
                for method, (lo, hi) in bounds.items():
                    for b, label in enumerate(X3_BIN_LABELS):
                        mask = bins == b
                        if mask.any():
                            cond_records.append(dict(
                                abs_x3=label, method=method,
                                **bounds_stats(lo[mask], hi[mask], y_test[mask]),
                            ))
    return _summarize(records, ["x3_shift", "method"]), _summarize(cond_records, ["method", "abs_x3"])


# ---------------------------------------------------------------------------
# Online / streaming shift: adaptive conformal inference
# ---------------------------------------------------------------------------

def stream_scenarios(T: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(x_loc path, noise_std path) for each streaming scenario."""
    ramp = np.linspace(0.0, 1.0, T)
    flat = np.zeros(T)
    return {
        "Gradual covariate drift": (2.0 * ramp, 1.0 + flat),
        "Gradual noise drift": (flat, 1.0 + 1.5 * ramp),
        "Sudden noise jump": (flat, np.where(np.arange(T) < T // 2, 1.0, 2.0)),
    }


def online_experiment(
    model, alpha: float, n_trials: int = N_TRIALS, T: int = 4000, n_calib: int = 500,
    gamma: float = 0.005, window: int = 500, local_window: int = 500, n_points: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream T points under each scenario, revealing each label after predicting.

    Returns
      summary   per (scenario, method): long-run coverage, worst local coverage
                (lowest coverage over any `local_window` consecutive steps, a
                measure of how badly a method can fail for a stretch of time),
                median interval length, share of infinite intervals
      series    per (scenario, method, t): coverage averaged over trials and a
                trailing `local_window` steps, and the median interval length
                over trials, thinned to `n_points` time points for plotting
    """
    scenarios = stream_scenarios(T)
    records, series = [], []
    for name, (loc_path, noise_path) in scenarios.items():
        cov_sum = {m: np.zeros(T) for m in ONLINE_METHODS}
        widths = {m: np.zeros((n_trials, T)) for m in ONLINE_METHODS}
        for r in range(n_trials):
            _, calib_scores, _ = _source_calibration(model, alpha, n_calib, r)
            X, y = make_stream((_STREAM, list(scenarios).index(name), r), loc_path, noise_path)
            stream_scores = np.abs(y - model.predict(X))
            out = run_online(calib_scores, stream_scores, alpha, gamma=gamma, window=window)
            for m, (covered, half_width) in out.items():
                cov_sum[m] += covered
                widths[m][r] = 2.0 * half_width
                local = np.convolve(covered, np.ones(local_window) / local_window, mode="valid")
                records.append(dict(
                    scenario=name, method=m,
                    coverage=covered.mean(), worst_local_coverage=local.min(),
                    median_width=float(np.median(2.0 * half_width)),
                    frac_infinite=float(np.isinf(half_width).mean()),
                ))

        idx = np.linspace(local_window - 1, T - 1, n_points).astype(int)
        for m in ONLINE_METHODS:
            mean_cov = cov_sum[m] / n_trials
            trailing = np.convolve(mean_cov, np.ones(local_window) / local_window, mode="full")[:T]
            trailing[: local_window - 1] = np.nan
            med_width = np.median(widths[m], axis=0)
            for t in idx:
                series.append(dict(scenario=name, method=m, t=int(t),
                                   local_coverage=trailing[t], median_width=med_width[t]))

    df = pd.DataFrame(records)
    summary = (
        df.groupby(["scenario", "method"], sort=False)
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            worst_local_coverage=("worst_local_coverage", "mean"),
            width_median=("median_width", "median"),
            frac_infinite=("frac_infinite", "mean"),
        )
        .reset_index()
    )
    return summary, pd.DataFrame(series)
