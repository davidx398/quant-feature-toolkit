# relative_momentum.py
from __future__ import annotations
import numpy as np
from .utils import safe_log, pct_change_log

def relative_momentum(
    close: np.ndarray,
    lookback: int = 60,
    k: int = 1,
    smooth: int | None = None,
) -> np.ndarray:
    """
    Simple JANUS-style relative momentum panel:
    - compute k-bar log returns for each market
    - trailing mean over `lookback`
    - (optional) simple smoothing with trailing mean over `smooth`

    Returns: array shape (n_markets, n_bars)
    """
    c = np.asarray(close, float)
    if c.ndim != 2:
        raise ValueError("close must be 2D: (n_markets, n_bars)")
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    r = pct_change_log(c, k=k)  # (M, N)
    # trailing mean over last `lookback` rows, per market
    kernel = np.ones(lookback, dtype=float) / lookback

    def _roll_mean(v):
        out = np.convolve(v, kernel, mode="valid")
        pad = np.full(lookback - 1, np.nan)
        return np.concatenate([pad, out])

    rm = np.apply_along_axis(_roll_mean, -1, r)  # (M, N)

    if smooth and smooth > 1:
        k2 = np.ones(smooth, dtype=float) / smooth
        def _roll2(v):
            out = np.convolve(v, k2, mode="same")
            return out
        rm = np.apply_along_axis(_roll2, -1, rm)

    return rm


def top_n_signal(rm: np.ndarray, top_n: int = 2) -> np.ndarray:
    """
    Long-only top-N selector from a panel `rm` (M, N).
    Returns a (M, N) mask of 1.0 for the top-N markets each bar, else 0.0.
    Ties are broken by NumPy argsort’s stable ordering.
    """
    x = np.asarray(rm, float)
    if x.ndim != 2:
        raise ValueError("rm must be 2D: (n_markets, n_bars)")
    M, N = x.shape
    top_n = int(max(1, min(top_n, M)))

    out = np.zeros_like(x, dtype=float)
    # argsort descending per column
    order = np.argsort(-x, axis=0)  # indices of markets sorted high→low
    for t in range(N):
        # pick first top_n rows at column t
        out[order[:top_n, t], t] = 1.0
    return out
