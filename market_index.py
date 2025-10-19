# market_index.py
from __future__ import annotations
import numpy as np
from .utils import pct_change_log, safe_log, normal_cdf

def compute_msi(close: np.ndarray, lookback: int = 60) -> np.ndarray:
    """
    Market Sentiment Index (simple breadth proxy):
    fraction of markets with positive lookback log-return, scaled to [-50,50]-ish.
    Guards all-NaN windows explicitly to avoid RuntimeWarning.
    """
    c = np.asarray(close, float)  # (M, N)
    if c.ndim != 2:
        raise ValueError("close must be (n_markets, n_bars)")
    M, N = c.shape
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    r = pct_change_log(c, k=lookback)  # (M, N)
    # breadth in [-1,1]
    pos_frac = np.nanmean(np.sign(r), axis=0)  # (N,)
    # mild compression to [-50,50]
    out = 100.0 * normal_cdf(2.0 * pos_frac) - 50.0

    # guard lead-in region where returns are NaN (first lookback bars): set to 0
    out[:lookback] = 0.0
    return out


def index_returns(close: np.ndarray, horizon: int = 5) -> np.ndarray:
    """
    Equal-weight index log-returns over `horizon`.
    """
    c = np.asarray(close, float)  # (M, N)
    idx = np.nanmean(c, axis=0)   # (N,)
    ret = np.zeros_like(idx)
    if horizon > 0 and idx.size > horizon:
        ret[horizon:] = np.log(np.maximum(idx[horizon:], 1e-30) / np.maximum(idx[:-horizon], 1e-30))
    return ret


def cma_oos_signal(series: np.ndarray, lookback: int = 20, win: int = 40) -> np.ndarray:
    """
    Very simple CMA OOS-style signal for a single series:
    sign(series - SMA(lookback)), then evaluated OOS by shifting forward `win`.
    """
    x = np.asarray(series, float).ravel()
    n = x.size
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    kernel = np.ones(lookback, dtype=float) / lookback
    sma = np.convolve(x, kernel, mode="same")
    sig = np.sign(x - sma)
    # out-of-sample shift (apply signal decided `win` bars ago)
    if win > 0:
        sig[:-win] = sig[win:]
        sig[-win:] = 0.0
    return sig
