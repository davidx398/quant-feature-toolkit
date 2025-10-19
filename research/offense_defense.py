# offense_defense.py
from __future__ import annotations
import numpy as np
from .utils import pct_change_log

def rs_panel(close: np.ndarray, lookback: int = 60) -> np.ndarray:
    """
    Relative Strength panel:
      RS_i(t) = k-day log return for market i minus the cross-sectional mean at t.
    close: (M, N) -> (M, N)
    """
    C = np.asarray(close, float)
    r = pct_change_log(C, lookback)            # (M,N)
    mu = np.nanmean(r, axis=0, keepdims=True)  # cross-sectional mean per bar
    return r - mu

def rs_fractiles(x: np.ndarray, qs=(0.2, 0.5, 0.8)):
    """
    Row-wise fractiles (per market across time) if x is (M, T),
    otherwise column-wise if x is (T,).
    Returns a tuple of arrays matching the non-quantile axes.
    """
    X = np.asarray(x, float)
    if X.ndim == 2:
        # by market across time
        out = tuple(np.quantile(X, q, axis=1) for q in qs)
    elif X.ndim == 1:
        out = tuple(np.quantile(X, q) for q in qs)
    else:
        raise ValueError("x must be 1D or 2D")
    return out

def offense_index(
    rs: np.ndarray,
    mode: str = "quantile",
    q: float = 0.60,
) -> np.ndarray:
    """
    Offense breadth at each bar: fraction of markets with high RS.
    """
    rs = np.asarray(rs, float)
    if mode == "zero":
        return (rs > 0).mean(axis=0)
    if mode == "median":
        thr = np.median(rs, axis=0, keepdims=True)
        return (rs >= thr).mean(axis=0)
    if mode == "quantile":
        thr = np.quantile(rs, q, axis=0, keepdims=True)
        return (rs >= thr).mean(axis=0)
    raise ValueError("mode must be 'quantile', 'median', or 'zero'")

def defense_index(
    rs: np.ndarray,
    mode: str = "quantile",
    q: float = 0.40,
) -> np.ndarray:
    """
    Defense breadth at each bar: fraction of markets with low RS.
    """
    rs = np.asarray(rs, float)
    if mode == "zero":
        return (rs < 0).mean(axis=0)
    if mode == "median":
        thr = np.median(rs, axis=0, keepdims=True)
        return (rs <= thr).mean(axis=0)
    if mode == "quantile":
        thr = np.quantile(rs, q, axis=0, keepdims=True)
        return (rs <= thr).mean(axis=0)
    raise ValueError("mode must be 'quantile', 'median', or 'zero'")
