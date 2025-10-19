# entropy_direction.py
from __future__ import annotations
import numpy as np
from .utils import pct_change_log

def doe(close: np.ndarray, lookback: int = 10, bins: int = 8) -> np.ndarray:
    """
    Optional entropy-flavored direction measure:
    - For each market & bar, histogram the last lookback returns into `bins`.
    - Compute Shannon entropy and how it changes; breadth the direction across markets.

    Returns a [0,1] breadth-like series where >0.5 suggests "entropy increasing".
    """
    C = np.asarray(close, float)
    M, N = C.shape
    r = pct_change_log(C, 1)  # daily log returns
    out = np.zeros(N, dtype=float)

    for t in range(lookback, N):
        ent = []
        prev = []
        for m in range(M):
            seg = r[m, t - lookback + 1 : t + 1]
            seg_prev = r[m, t - lookback : t]  # one step earlier
            # hist on common edges for stability
            lo = np.nanpercentile(seg, 2)
            hi = np.nanpercentile(seg, 98)
            edges = np.linspace(lo, hi, bins + 1) if hi > lo else np.linspace(-1, 1, bins + 1)
            p,_ = np.histogram(seg, bins=edges, density=True)
            q,_ = np.histogram(seg_prev, bins=edges, density=True)
            p = p / (p.sum() + 1e-12)
            q = q / (q.sum() + 1e-12)
            Hp = -(p * np.log(p + 1e-12)).sum()
            Hq = -(q * np.log(q + 1e-12)).sum()
            ent.append(Hp); prev.append(Hq)
        ent = np.array(ent); prev = np.array(prev)
        out[t] = (ent > prev).mean()  # breadth >0.5 → entropy rising
    return out
