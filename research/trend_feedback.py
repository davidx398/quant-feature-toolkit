# trend_feedback.py
from __future__ import annotations
import numpy as np
from .utils import pct_change_log

def rss(close: np.ndarray, lookback: int = 60, top_q: float = 0.8, bot_q: float = 0.2) -> np.ndarray:
    """
    Relative Strength Spread = mean RS of leaders - mean RS of laggards (at each bar).
    We compute RS as k-day log return minus cross-sectional mean at each bar.
    """
    C = np.asarray(close, float)
    if C.ndim != 2:
        raise ValueError("close must be 2D")
    r = pct_change_log(C, lookback)            # (M,N)
    mu = np.nanmean(r, axis=0, keepdims=True)
    rs = r - mu

    # thresholds per bar
    t_top = np.quantile(rs, top_q, axis=0, keepdims=True)
    t_bot = np.quantile(rs, bot_q, axis=0, keepdims=True)

    leaders = np.where(rs >= t_top, rs, np.nan)
    laggards = np.where(rs <= t_bot, rs, np.nan)

    mean_lead = np.nanmean(leaders, axis=0)
    mean_lagg = np.nanmean(laggards, axis=0)
    return mean_lead - mean_lagg

def drss(rss_series: np.ndarray, k: int = 10) -> np.ndarray:
    """Change in RSS over k bars."""
    x = np.asarray(rss_series, float)
    out = np.zeros_like(x)
    if k > 0:
        out[k:] = x[k:] - x[:-k]
    return out
