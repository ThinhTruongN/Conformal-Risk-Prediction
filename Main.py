"""
Conformal Prediction Under Distribution Shift
Thinh Truong Nguyen

Runs the full experiment suite end-to-end:
  1. baseline split-conformal calibration (no shift)
  2. covariate / noise / subgroup shift sweeps
  3. attempted fixes: recalibration, weighted conformal (exact and estimated
     likelihood ratios), group-specific calibration, larger calibration sets,
     alternative regression models
  4. locally adaptive intervals (normalized scores, CQR) on heteroscedastic data
  5. online prediction on drifting streams: adaptive conformal inference (ACI)
     and rolling-window calibration
Every shift experiment is repeated over N_TRIALS independent calibration/test
draws and reports mean coverage with its trial-to-trial spread.
Saves every figure and results table under results/.
"""

from __future__ import annotations

import os
import time

import pandas as pd

from experiments import (
    N_TRIALS,
    adaptive_width_experiment,
    calibration_size_experiment,
    covariate_shift_experiment,
    model_comparison_experiment,
    noise_shift_experiment,
    online_experiment,
    run_baseline,
    subgroup_shift_experiment,
)
from plots import (
    plot_baseline_calibration,
    plot_calibration_size,
    plot_conditional_coverage,
    plot_method_comparison,
    plot_online,
    plot_shift_curve,
)

RESULTS_DIR = "results"
ALPHA = 0.10
TARGET_COVERAGE = 1 - ALPHA
SEED = 0
ONLINE_LOCAL_WINDOW = 500

pd.set_option("display.float_format", lambda v: f"{v:.3f}")
pd.set_option("display.width", 140)

SUMMARY_COLS = ["coverage_mean", "coverage_std", "coverage_se", "width_mean"]


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def only(df: pd.DataFrame, method: str) -> pd.DataFrame:
    return df[df["method"] == method].drop(columns="method").reset_index(drop=True)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    start = time.time()

    print("=" * 70)
    print("Conformal Prediction Under Distribution Shift")
    print(f"({N_TRIALS} independent calibration/test draws per setting)")
    print("=" * 70)

    # 1. Baseline: fit model, calibrate, check coverage on one unshifted test set.
    baseline = run_baseline(alpha=ALPHA, seed=SEED)
    model, width = baseline["model"], baseline["width"]

    section("1. Baseline (no shift, single illustrative draw)")
    print(f"Target coverage:        {TARGET_COVERAGE:.1%}")
    print(f"Conformal half-width:   +/- {width:.3f}")
    print(f"Empirical coverage:     {baseline['coverage']:.1%}")
    print(f"Average interval length: {baseline['avg_width']:.3f}")
    plot_baseline_calibration(
        model, baseline["X_test"], baseline["y_test"], width, baseline["coverage"],
        f"{RESULTS_DIR}/baseline.png",
    )

    # 2. Covariate shift: standard conformal, plus all covariate-shift fixes on shared test sets.
    cov_shifts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    cov_df = covariate_shift_experiment(model, ALPHA, cov_shifts)
    cov_std = only(cov_df, "standard")
    cov_std.to_csv(f"{RESULTS_DIR}/covariate_shift.csv", index=False)
    plot_shift_curve(
        cov_std, "shift", "Covariate shift (mean of x)", "Coverage under covariate shift",
        TARGET_COVERAGE, f"{RESULTS_DIR}/covariate_shift.png",
    )
    section("2. Covariate shift sweep (standard conformal)")
    print(f"Repeated-trial baseline (shift = 0): {cov_std.loc[0, 'coverage_mean']:.1%} "
          f"± {cov_std.loc[0, 'coverage_std']:.1%} (sd)")
    print(cov_std[["shift"] + SUMMARY_COLS].to_string(index=False))

    # 3. Noise shift sweep.
    noise_levels = [1.0, 1.5, 2.0, 2.5, 3.0]
    noise_df = only(noise_shift_experiment(model, ALPHA, noise_levels), "standard")
    noise_df.to_csv(f"{RESULTS_DIR}/noise_shift.csv", index=False)
    plot_shift_curve(
        noise_df, "noise_std", "Noise std of test distribution", "Coverage under noise shift",
        TARGET_COVERAGE, f"{RESULTS_DIR}/noise_shift.png",
    )
    section("3. Noise shift sweep")
    print(noise_df[["noise_std"] + SUMMARY_COLS].to_string(index=False))

    # 4. Subgroup shift: calibrated on the baseline mixture (20% group B), so
    #    frac_b = 0.2 is the no-shift point.
    frac_bs = [0.2, 0.35, 0.5, 0.65, 0.8]
    sub_df = subgroup_shift_experiment(model, ALPHA, frac_bs, baseline_frac_b=0.2)
    sub_std = only(sub_df, "standard")
    sub_std.to_csv(f"{RESULTS_DIR}/subgroup_shift.csv", index=False)
    plot_shift_curve(
        sub_std, "frac_b", "Fraction of test set from high-noise group B",
        "Coverage under subgroup shift (calibrated at 20% B)",
        TARGET_COVERAGE, f"{RESULTS_DIR}/subgroup_shift.png",
    )
    section("4. Subgroup shift sweep (calibrated on 20% group B)")
    print(sub_std[["frac_b"] + SUMMARY_COLS].to_string(index=False))

    # 5. Fixes under covariate shift: recalibration vs. weighted conformal.
    cov_df.to_csv(f"{RESULTS_DIR}/fixes_covariate.csv", index=False)
    plot_method_comparison(
        cov_df, "shift", "method", "Covariate shift (mean of x)",
        "Fixes under covariate shift", TARGET_COVERAGE, f"{RESULTS_DIR}/fixes_covariate.png",
        width_col="width_median", width_label="Median interval length (log scale)",
        log_width=True, show_infinite=True,
    )
    section("5. Fixes under covariate shift")
    print(cov_df[["shift", "method", "coverage_mean", "coverage_std", "width_median", "frac_infinite"]]
          .to_string(index=False))

    # 6. Fix: group-specific calibration under subgroup (mixture) shift.
    sub_df.to_csv(f"{RESULTS_DIR}/fixes_subgroup.csv", index=False)
    plot_method_comparison(
        sub_df, "frac_b", "method", "Fraction of test set from high-noise group B",
        "Group-specific calibration under subgroup shift", TARGET_COVERAGE,
        f"{RESULTS_DIR}/fixes_subgroup.png",
    )
    section("6. Fix: group-specific calibration under subgroup shift")
    print(sub_df[["frac_b", "method"] + SUMMARY_COLS].to_string(index=False))

    # 7. Does a larger (shifted-distribution) calibration set stabilize recalibration?
    calib_sizes = [200, 500, 1000]
    calib_df = calibration_size_experiment(model, ALPHA, calib_sizes, shift=2.0)
    calib_df.to_csv(f"{RESULTS_DIR}/calibration_size.csv", index=False)
    plot_calibration_size(calib_df, TARGET_COVERAGE, f"{RESULTS_DIR}/calibration_size.png")
    section("7. Fix: calibration set size (recalibrating at covariate shift = 2.0)")
    print(calib_df[["calib_size", "coverage_mean", "coverage_std", "width_mean", "width_std"]]
          .to_string(index=False))

    # 8. Does the choice of base model matter under noise shift?
    model_df = model_comparison_experiment(ALPHA, noise_levels)
    model_df.to_csv(f"{RESULTS_DIR}/method_comparison.csv", index=False)
    plot_method_comparison(
        model_df, "noise_std", "model", "Noise std of test distribution",
        "Base-model comparison under noise shift", TARGET_COVERAGE,
        f"{RESULTS_DIR}/method_comparison.png",
    )
    section("8. Model comparison under noise shift")
    print(model_df[["model", "noise_std"] + SUMMARY_COLS].to_string(index=False))

    # 9. Locally adaptive intervals on heteroscedastic data (noise std = 0.2 + |x3|),
    #    under a shift of x3 toward the noisy region.
    x3_shifts = [0.0, 0.5, 1.0, 1.5, 2.0]
    adapt_df, cond_df = adaptive_width_experiment(ALPHA, x3_shifts)
    adapt_df.to_csv(f"{RESULTS_DIR}/adaptive_shift.csv", index=False)
    cond_df.to_csv(f"{RESULTS_DIR}/adaptive_conditional.csv", index=False)
    plot_method_comparison(
        adapt_df, "x3_shift", "method", "Shift in mean of x3 (toward high noise)",
        "Locally adaptive intervals under heteroscedastic shift", TARGET_COVERAGE,
        f"{RESULTS_DIR}/adaptive_shift.png",
    )
    plot_conditional_coverage(cond_df, TARGET_COVERAGE, f"{RESULTS_DIR}/adaptive_conditional.png")
    section("9. Locally adaptive intervals (heteroscedastic data, shift in x3)")
    print(adapt_df[["x3_shift", "method"] + SUMMARY_COLS].to_string(index=False))
    print()
    print("Conditional coverage by |x3| at no shift:")
    print(cond_df[["method", "abs_x3", "coverage_mean", "coverage_std", "width_mean"]].to_string(index=False))

    # 10. Online prediction on drifting streams: ACI vs. rolling-window calibration.
    online_df, online_series = online_experiment(model, ALPHA, local_window=ONLINE_LOCAL_WINDOW)
    online_df.to_csv(f"{RESULTS_DIR}/online_summary.csv", index=False)
    online_series.to_csv(f"{RESULTS_DIR}/online_series.csv", index=False)
    plot_online(online_series, TARGET_COVERAGE, ONLINE_LOCAL_WINDOW, f"{RESULTS_DIR}/online.png")
    section("10. Online conformal prediction (ACI, gamma = 0.005; rolling window = 500)")
    print(online_df.to_string(index=False))

    print()
    print(f"All figures and tables written to {RESULTS_DIR}/ ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
