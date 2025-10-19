# direction_momentum.py
from __future__ import annotations
import numpy as np
from .utils import pct_change_log

def dom_index(close: np.ndarray, lookback: int = 10) -> np.ndarray:
    """
    Direction of Momentum (breadth):
      fraction of markets with k-day return > 0 at each bar. Range [0,1].
    """
    C = np.asarray(close, float)
    r = pct_change_log(C, lookback)
    return (r > 0).mean(axis=0)

def doe_index(close: np.ndarray, lookback: int = 10) -> np.ndarray:
    """
    Direction of 'Entropy' (magnitude-weighted breadth):
      average of sign(k-day return) * normalized magnitude.
      Scales to [0,1] by mapping mean in [-1,1] → (x+1)/2.
    """
    C = np.asarray(close, float)
    r = pct_change_log(C, lookback)     # (M,N)
    mag = np.nanmedian(np.abs(r), axis=0, keepdims=True) + 1e-12
    score = np.sign(r) * (np.abs(r) / mag)          # robust magnitude weighting
    m = np.nanmean(score, axis=0)                   # in ~[-?, ?]
    return 0.5 * (np.clip(m, -1.0, 1.0) + 1.0)      # [0,1]
