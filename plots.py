"""
Figures for the report. All plots are static PNGs (matplotlib, Agg backend).

Palette (validated colorblind-safe, see project notes):
  blue   #2a78d6  - primary series / "fixed" state
  red    #e34948  - "standard / uncorrected" state
  aqua   #1baf7a  - categorical series 2
  yellow #eda100  - categorical series 3
  gray   #898781  - muted reference lines (target coverage)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd

BLUE = "#2a78d6"
RED = "#e34948"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GRAY = "#898781"
GRID = "#e1e0d9"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.edgecolor": "#c3c2b7",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _target_line(ax, target: float) -> None:
    ax.axhline(target, color=GRAY, linestyle="--", linewidth=1.5, zorder=1)
    ax.annotate(
        f"target coverage ({target:.0%})", xy=(1.0, target), xycoords=("axes fraction", "data"),
        xytext=(-4, 6), textcoords="offset points", ha="right", va="bottom", color=GRAY, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85), zorder=5,
    )


def plot_baseline_calibration(model, X_test, y_test, width, coverage, savepath):
    """Predicted vs actual with the conformal band around the diagonal."""
    pred = model.predict(X_test)
    covered = (y_test >= pred - width) & (y_test <= pred + width)

    lo, hi = min(pred.min(), y_test.min()), max(pred.max(), y_test.max())
    pad = 0.05 * (hi - lo)
    line_x = np.linspace(lo - pad, hi + pad, 100)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.fill_between(line_x, line_x - width, line_x + width, color=BLUE, alpha=0.12, zorder=1, label="90% interval band")
    ax.plot(line_x, line_x, color=GRAY, linewidth=1.5, zorder=2, label="perfect prediction")
    ax.scatter(pred[covered], y_test[covered], s=14, color=BLUE, alpha=0.6, zorder=3, label="covered")
    ax.scatter(pred[~covered], y_test[~covered], s=18, color=RED, alpha=0.8, zorder=4, marker="x", label="not covered")

    ax.set_xlabel("Predicted y")
    ax.set_ylabel("Actual y")
    ax.set_title(f"Baseline calibration (no shift) — empirical coverage {coverage:.1%}")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


# Fixed visual identity per method/model: color, marker, and line style, so
# identity never relies on color alone. Order is the categorical order.
STYLES = {
    "standard": dict(color=RED, marker="x", linestyle="--", label="Standard (no correction)"),
    "recalibrated": dict(color=BLUE, marker="o", linestyle="-", label="Recalibrated (new labels)"),
    "weighted_oracle": dict(color=AQUA, marker="s", linestyle="-.", label="Weighted, exact weights"),
    "weighted_estimated": dict(color=YELLOW, marker="^", linestyle=":", label="Weighted, estimated weights"),
    "group_specific": dict(color=BLUE, marker="o", linestyle="-", label="Group-specific"),
    "normalized": dict(color=BLUE, marker="o", linestyle="-", label="Normalized score"),
    "cqr": dict(color=AQUA, marker="s", linestyle="-.", label="CQR"),
    "aci": dict(color=YELLOW, marker="^", linestyle=":", label="ACI (fixed scores)"),
    "rolling": dict(color=AQUA, marker="s", linestyle="-.", label="Rolling window"),
    "rolling_aci": dict(color=BLUE, marker="o", linestyle="-", label="Rolling window + ACI"),
    "Linear Regression": dict(color=BLUE, marker="o", linestyle="-", label="Linear Regression"),
    "Random Forest": dict(color=AQUA, marker="s", linestyle="--", label="Random Forest"),
    "Gradient Boosting": dict(color=YELLOW, marker="^", linestyle=":", label="Gradient Boosting"),
}


COVERAGE_LABEL = "Empirical coverage\n(mean ± 1 sd over trials)"
SINGLE_SERIES = dict(color=BLUE, marker="o", linestyle="-", label=None)


def _plot_series(ax, x, y, style, band=None):
    """Line + markers, with an optional shaded ±1 sd band. Infinite values are not drawn."""
    y = np.where(np.isfinite(y), y, np.nan)
    if band is not None:
        ax.fill_between(x, y - band, y + band, color=style["color"], alpha=0.12, linewidth=0, zorder=2)
    ax.plot(x, y, color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
            markersize=8, linewidth=2, label=style["label"], zorder=3)


def plot_shift_curve(df: pd.DataFrame, x_col: str, xlabel: str, title: str, target: float, savepath: str):
    """Coverage of standard conformal vs. shift strength, with a ±1 sd band."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    _plot_series(ax, df[x_col], df["coverage_mean"], SINGLE_SERIES, band=df["coverage_std"])
    _target_line(ax, target)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(COVERAGE_LABEL)
    ax.set_title(title)
    ax.set_ylim(min(0.5, (df["coverage_mean"] - df["coverage_std"]).min() - 0.05), 1.02)
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def plot_method_comparison(
    df: pd.DataFrame, x_col: str, method_col: str, xlabel: str, title: str, target: float, savepath: str,
    width_col: str = "width_mean", width_label: str = "Average interval length",
    log_width: bool = False, show_infinite: bool = False,
):
    """Side-by-side panels: coverage (±1 sd band), interval length, and optionally
    the share of infinite intervals. One axis per panel, shared x."""
    n_panels = 3 if show_infinite else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.8))
    methods = list(dict.fromkeys(df[method_col]))

    for m in methods:
        g = df[df[method_col] == m]
        style = STYLES[m]
        _plot_series(axes[0], g[x_col], g["coverage_mean"], style, band=g["coverage_std"])
        _plot_series(axes[1], g[x_col], g[width_col], style)
        if show_infinite:
            _plot_series(axes[2], g[x_col], g["frac_infinite"], style)

    _target_line(axes[0], target)
    low = (df["coverage_mean"] - df["coverage_std"]).min()
    axes[0].set_ylim(min(0.5, low - 0.05), 1.02)
    axes[0].set_ylabel(COVERAGE_LABEL)
    axes[0].set_title("Coverage")
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)

    axes[1].set_ylabel(width_label)
    axes[1].set_title("Interval length")
    if log_width:
        axes[1].set_yscale("log")
        axes[1].yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        axes[1].yaxis.set_major_formatter(ScalarFormatter())
        axes[1].yaxis.set_minor_formatter(NullFormatter())

    if show_infinite:
        axes[2].set_ylabel("Share of test points")
        axes[2].set_title("Infinite (uninformative) intervals")
        axes[2].set_ylim(-0.03, 1.03)

    for ax in axes:
        ax.set_xlabel(xlabel)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def plot_calibration_size(df: pd.DataFrame, target: float, savepath: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(
        df["calib_size"], df["coverage_mean"], yerr=df["coverage_std"],
        color=BLUE, marker="o", markersize=8, linewidth=2, capsize=4, zorder=3,
    )
    _target_line(ax, target)
    ax.set_xlabel("Calibration set size (drawn from the shifted distribution)")
    ax.set_ylabel(COVERAGE_LABEL)
    ax.set_title("Recalibration reliability vs. calibration set size")
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(df["calib_size"])
    ax.set_xticklabels(df["calib_size"])
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def plot_conditional_coverage(df: pd.DataFrame, target: float, savepath: str):
    """Coverage and interval length within bins of |x3| (the noise driver), at no shift."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = list(dict.fromkeys(df["abs_x3"]))
    x = np.arange(len(labels))
    for m in dict.fromkeys(df["method"]):
        g = df[df["method"] == m].set_index("abs_x3").loc[labels]
        _plot_series(axes[0], x, g["coverage_mean"].to_numpy(), STYLES[m], band=g["coverage_std"].to_numpy())
        _plot_series(axes[1], x, g["width_mean"].to_numpy(), STYLES[m])

    _target_line(axes[0], target)
    low = (df["coverage_mean"] - df["coverage_std"]).min()
    axes[0].set_ylim(min(0.4, low - 0.05), 1.02)
    axes[0].set_ylabel("Coverage within bin\n(mean ± 1 sd over trials)")
    axes[0].set_title("Conditional coverage")
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    axes[1].set_ylabel("Average interval length")
    axes[1].set_title("Interval length")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("|x3|  (noise std = 0.2 + |x3|)")
    fig.suptitle("Coverage by local difficulty, no shift", fontsize=13)
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def plot_online(series: pd.DataFrame, target: float, local_window: int, savepath: str):
    """Grid: one column per streaming scenario; top row trailing-window coverage,
    bottom row median interval length, both over time."""
    scenarios = list(dict.fromkeys(series["scenario"]))
    fig, axes = plt.subplots(2, len(scenarios), figsize=(6 * len(scenarios), 8.5), sharex=True)
    for j, sc in enumerate(scenarios):
        for m in dict.fromkeys(series["method"]):
            g = series[(series["scenario"] == sc) & (series["method"] == m)]
            style = STYLES[m] | dict(marker=None)
            _plot_series(axes[0, j], g["t"].to_numpy(), g["local_coverage"].to_numpy(), style)
            _plot_series(axes[1, j], g["t"].to_numpy(), g["median_width"].to_numpy(), style)
        _target_line(axes[0, j], target)
        axes[0, j].set_title(sc)
        axes[0, j].set_ylim(0.45, 1.0)
        axes[1, j].set_xlabel("Time step")
    axes[0, 0].set_ylabel(f"Coverage over last {local_window} steps\n(averaged over trials)")
    axes[1, 0].set_ylabel("Median interval length\n(across trials)")
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=9)
    fig.suptitle("Online conformal prediction under drifting data", fontsize=13)
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)
