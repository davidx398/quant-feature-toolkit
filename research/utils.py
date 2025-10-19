# janus_indicators/utils.py
from __future__ import annotations
import math
import numpy as np

EPS = 1e-30

def normal_cdf(x):
    """
    Standard normal CDF. Works with scalars or ndarrays.
    Uses math.erf elementwise (no SciPy dependency).
    """
    x = np.asarray(x, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))

def safe_log(x):
    x = np.asarray(x, dtype=float)
    return np.log(np.maximum(x, EPS))

def pct_change_log(x: np.ndarray, k: int = 1) -> np.ndarray:
    """k-bar log return: log(x_t/x_{t-k}). Front k values = 0."""
    x = np.asarray(x, float)
    out = np.zeros_like(x, dtype=float)
    if k <= 0:
        return out
    out[..., k:] = safe_log(x[..., k:] / np.maximum(x[..., :-k], EPS))
    return out

def sma(x: np.ndarray, win: int) -> np.ndarray:
    """Simple moving average with NaN lead-in to preserve alignment."""
    x = np.asarray(x, float)
    if win <= 1:
        return x.copy()
    kernel = np.ones(win, dtype=float) / win
    out_valid = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), -1, x)
    pad_shape = list(x.shape[:-1]) + [win - 1]
    pad = np.full(pad_shape, np.nan, dtype=float)
    return np.concatenate([pad, out_valid], axis=-1)

def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Classic EMA. alpha in (0,1]."""
    x = np.asarray(x, float)
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0,1]")
    out = np.empty_like(x, dtype=float)
    out[..., 0] = x[..., 0]
    for i in range(1, x.shape[-1]):
        out[..., i] = alpha * x[..., i] + (1.0 - alpha) * out[..., i - 1]
    return out

def trailing_mean(x: np.ndarray, win: int) -> np.ndarray:
    """Trailing mean with NaN lead-in."""
    return sma(x, win)

def trailing_std(x: np.ndarray, win: int) -> np.ndarray:
    """Trailing std (population) with NaN lead-in."""
    x = np.asarray(x, float)
    if win <= 1:
        return np.zeros_like(x)
    kernel = np.ones(win, dtype=float) / win
    def _roll_std(v):
        m = np.convolve(v, kernel, mode="valid")
        m2 = np.convolve(v*v, kernel, mode="valid")
        var = np.maximum(m2 - m*m, 0.0)
        return np.sqrt(var)
    out_valid = np.apply_along_axis(_roll_std, -1, x)
    pad_shape = list(x.shape[:-1]) + [win - 1]
    pad = np.full(pad_shape, np.nan, dtype=float)
    return np.concatenate([pad, out_valid], axis=-1)

# ---------- missing helpers that core.py expects ----------

def median_axis0(x: np.ndarray) -> np.ndarray:
    """
    Median along axis=0 (column-wise for shape (n_markets, n_bars)).
    Uses np.nanmedian if you want to ignore NaNs; here we use np.median
    to match Masters’ usual “no NaN” path.
    """
    x = np.asarray(x, float)
    return np.median(x, axis=0)

def argsort(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Stable argsort wrapper (mergesort). Some ranking logic benefits from stability.
    """
    return np.argsort(a, axis=axis, kind="mergesort")
