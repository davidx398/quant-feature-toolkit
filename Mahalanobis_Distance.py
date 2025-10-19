#!/usr/bin/env python3
"""
Mahalanobis multiple-market indicator (Masters, VAR_MAHAL)

- Window: lookback bars ending at icase-1 (current bar excluded)
- For each market i:
    μ_i = mean of log returns over that window
         = log(C[i, icase-1] / C[i, icase-lookback]) / (lookback-1)
- Covariance Σ from demeaned log returns over the window (length L=lookback-1)
- Squared Mahalanobis distance for the *current* bar's demeaned return vector:
      D2 = d^T Σ^{-1} d
- F-transform (Masters’ exact scaling + clamping) then logit:
      k = lookback-1-n_markets
      F = D2 * (lookback-1)*k / ( n_markets*(lookback-2)*lookback )
      p = F_CDF(ndf1=n_markets, ndf2=k, F)
      p = min(0.99999, max(0.5, p))
      out = log( p / (1-p) )
- Optional EMA smoothing over the output if n_to_smooth > 1.
"""

from __future__ import annotations
import math
from typing import Iterable
import numpy as np

EPS = 1e-30

# ---------- F CDF (pure Python; same formula Masters uses) ----------
def _betacf(a: float, b: float, x: float, max_iter: int = 200, tol: float = 3e-14) -> float:
    am, bm, az = 1.0, 1.0, 1.0
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    bz = 1.0 - qab * x / qap
    for m in range(1, max_iter + 1):
        em = float(m)
        tem = em + em
        d = em * (b - em) * x / ((qam + tem) * (a + tem))
        ap = az + d * am
        bp = bz + d * bm
        d = -(a + em) * (qab + em) * x / ((a + tem) * (qap + tem))
        app = ap + d * az
        bpp = bp + d * bz
        am, bm = ap / bpp, bp / bpp
        aold, az, bz = az, app / bpp, 1.0
        if abs(az - aold) < tol * abs(az):
            break
    return az

def _betainc_reg(a: float, b: float, x: float) -> float:
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    ln_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    else:
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b

def f_cdf(ndf1: int, ndf2: int, F: float) -> float:
    """CDF of F_{ndf1, ndf2}(F) = 1 - I_{ndf2/(ndf2 + ndf1*F)}(ndf2/2, ndf1/2)."""
    if F <= 0.0 or ndf1 <= 0 or ndf2 <= 0:
        return 0.0
    x = ndf2 / (ndf2 + ndf1 * F)
    p = 1.0 - _betainc_reg(0.5 * ndf2, 0.5 * ndf1, x)
    return 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)

# ---------- Mahalanobis indicator ----------
def mahal_indicator(
    close: Iterable[Iterable[float]],
    lookback: int,
    n_to_smooth: int = 1
) -> np.ndarray:
    """
    Parameters
    ----------
    close : array-like shape (n_markets, n_bars)
    lookback : int  (must be >= 3 and typically > n_markets+1)
    n_to_smooth : EMA length applied to the final series (Masters: 2/(N+1))

    Returns
    -------
    np.ndarray length n_bars (leading lookback bars set to 0.0)
    """
    C = np.asarray(close, dtype=float)
    if C.ndim != 2:
        raise ValueError("close must be 2D (n_markets, n_bars).")
    M, N = C.shape
    if lookback < 3:
        raise ValueError("lookback must be >= 3.")
    out = np.zeros(N, dtype=float)

    front_bad = lookback  # current bar is excluded from the window
    if N <= front_bad:
        return out

    L = lookback - 1  # number of returns in the window

    for t in range(front_bad, N):
        # mean of log-returns per market over the window [t-lookback .. t-1]
        mu = np.log(C[:, t-1] / np.maximum(C[:, t-lookback], EPS)) / L  # shape (M,)

        # Build demeaned return matrix R_demean: shape (M, L)
        # columns k=1..L correspond to returns at t-k
        numer = []
        for k in range(1, lookback):
            r_k = np.log(np.maximum(C[:, t-k], EPS) / np.maximum(C[:, t-k-1], EPS))  # (M,)
            numer.append(r_k - mu)
        R = np.column_stack(numer)  # (M, L)

        # Sample covariance (Masters divides by L)
        Sigma = (R @ R.T) / L  # (M, M)

        # Invert; if singular/ill-conditioned, output 0.0 for this t (as in C++)
        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            out[t] = 0.0
            continue

        # Current demeaned return vector
        d_now = np.log(np.maximum(C[:, t], EPS) / np.maximum(C[:, t-1], EPS)) - mu  # (M,)
        D2 = float(d_now.T @ Sigma_inv @ d_now)  # squared Mahalanobis distance

        # Masters’ F-transform and logit
        k_df2 = lookback - 1 - M
        if k_df2 <= 0:
            out[t] = 0.0
            continue
        F_val = D2 * (lookback - 1) * k_df2 / (M * (lookback - 2) * lookback)
        p = f_cdf(M, k_df2, F_val)
        p = 0.99999 if p > 0.99999 else (0.5 if p < 0.5 else p)
        out[t] = math.log(p / (1.0 - p))

    # Optional EMA smoothing
    if n_to_smooth and n_to_smooth > 1:
        alpha = 2.0 / (n_to_smooth + 1.0)
        for t in range(front_bad + 1, N):
            out[t] = alpha * out[t] + (1.0 - alpha) * out[t-1]

    return out

# ---------------- Demo ----------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    M, N = 5, 300
    # Make some correlated price paths
    base = np.cumsum(0.001 + 0.01 * rng.standard_normal(N))
    close = []
    for m in range(M):
        noise = 0.005 * rng.standard_normal(N)
        path = 100.0 * np.exp(base + noise + 0.0005 * m)
        close.append(path)
    close = np.vstack(close)

    lookback = 40
    n_to_smooth = 10

    vals = mahal_indicator(close, lookback, n_to_smooth)
    np.set_printoptions(precision=4, suppress=True)
    print("Mahalanobis (tail):", vals[-10:])
