"""
Synthetic data generation for the conformal-prediction-under-shift experiments.

The ground-truth rule is known because we generate the data ourselves:

    y = 2*x0 - 1.5*x1 + 0.8*x2**2 + sin(2*x3) + noise,  noise ~ Normal(0, noise_std)

Three kinds of distribution shift are supported, matching the experiment plan:
  - covariate shift: shift the mean of the input distribution (x_loc)
  - noise shift: increase the label noise std (noise_std)
  - subgroup shift: change the mixture proportion between a low-noise group A
    and a high-noise group B

Two further variants support the adaptive-method experiments:
  - heteroscedastic noise (hetero=True): noise std = noise_std * (0.2 + |x3|),
    so difficulty is visible in the inputs, unlike the latent subgroups above
  - streams (make_stream): x_loc and noise_std vary over time, one value per step
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

N_FEATURES = 4

# A seed is either an int or a tuple of ints, e.g. (experiment_id, level, trial).
# numpy's default_rng turns a tuple into an independent stream via SeedSequence,
# which is how the repeated-trial experiments get reproducible, non-overlapping draws.
Seed = int | Sequence[int]


def true_function(X: np.ndarray) -> np.ndarray:
    """The noiseless data-generating rule y = f(x)."""
    x0, x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return 2.0 * x0 - 1.5 * x1 + 0.8 * x2**2 + np.sin(2.0 * x3)


def hetero_noise_scale(X: np.ndarray) -> np.ndarray:
    """Relative noise std for the heteroscedastic variant: 0.2 + |x3|."""
    return 0.2 + np.abs(X[:, 3])


def make_data(
    n: int,
    seed: Seed,
    x_loc: float | np.ndarray = 0.0,
    x_scale: float = 1.0,
    noise_std: float = 1.0,
    hetero: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw n i.i.d. samples (X, y) from the synthetic model.

    x_loc / x_scale control covariate shift, noise_std controls noise shift.
    x_loc may be a length-N_FEATURES array to shift individual features.
    With hetero=True the noise std is noise_std * hetero_noise_scale(x).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=x_loc, scale=x_scale, size=(n, N_FEATURES))
    if hetero:
        noise = rng.normal(0.0, 1.0, size=n) * noise_std * hetero_noise_scale(X)
    else:
        noise = rng.normal(0.0, noise_std, size=n)
    return X, true_function(X) + noise


def make_stream(
    seed: Seed, x_loc_path: np.ndarray, noise_std_path: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a time-ordered stream: at step t, x_t ~ N(x_loc_path[t], I) and
    y_t = f(x_t) + Normal(0, noise_std_path[t]). The two paths set the length."""
    x_loc_path = np.asarray(x_loc_path, dtype=float)
    noise_std_path = np.asarray(noise_std_path, dtype=float)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(len(x_loc_path), N_FEATURES)) + x_loc_path[:, None]
    y = true_function(X) + rng.normal(size=len(X)) * noise_std_path
    return X, y


def make_subgroup_data(
    n: int,
    seed: Seed,
    frac_b: float,
    noise_std_a: float = 1.0,
    noise_std_b: float = 2.5,
    x_loc: float = 0.0,
    x_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a mixture of a low-noise group A and a high-noise group B.

    frac_b is the fraction of the sample coming from group B. The per-group
    conditional distributions never change; only the mixture weight does.
    Returns (X, y, group) where group is an array of 'A'/'B' labels.
    """
    rng = np.random.default_rng(seed)
    n_b = int(round(n * frac_b))
    n_a = n - n_b

    X_a = rng.normal(loc=x_loc, scale=x_scale, size=(n_a, N_FEATURES))
    y_a = true_function(X_a) + rng.normal(0.0, noise_std_a, size=n_a)

    X_b = rng.normal(loc=x_loc, scale=x_scale, size=(n_b, N_FEATURES))
    y_b = true_function(X_b) + rng.normal(0.0, noise_std_b, size=n_b)

    X = np.vstack([X_a, X_b])
    y = np.concatenate([y_a, y_b])
    group = np.array(["A"] * n_a + ["B"] * n_b)

    perm = rng.permutation(len(y))
    return X[perm], y[perm], group[perm]


def covariate_shift_log_weight(
    X: np.ndarray, x_loc: float, base_loc: float = 0.0, x_scale: float = 1.0
) -> np.ndarray:
    """Exact log likelihood ratio log[p_test(x) / p_calib(x)] for covariate shift.

    Calibration inputs are N(base_loc * 1, x_scale^2 I) and test inputs are
    N(x_loc * 1, x_scale^2 I), so with mu = (x_loc - base_loc) * 1:
        log w(x) = mu . (x - base_loc * 1) / x_scale^2 - |mu|^2 / (2 x_scale^2).
    These are the "oracle" weights for weighted conformal prediction.
    """
    delta = x_loc - base_loc
    d = X.shape[1]
    return (delta * (X - base_loc).sum(axis=1) - 0.5 * d * delta**2) / x_scale**2


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_calib: np.ndarray
    y_calib: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def train_calib_test_split(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    train_frac: float = 0.5,
    calib_frac: float = 0.25,
) -> Split:
    """Randomly split (X, y) into train / calibration / test sets."""
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    train_end = int(train_frac * n)
    calib_end = train_end + int(calib_frac * n)

    train_idx, calib_idx, test_idx = perm[:train_end], perm[train_end:calib_end], perm[calib_end:]
    return Split(
        X_train=X[train_idx], y_train=y[train_idx],
        X_calib=X[calib_idx], y_calib=y[calib_idx],
        X_test=X[test_idx], y_test=y[test_idx],
    )
