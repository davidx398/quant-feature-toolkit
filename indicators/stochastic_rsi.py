#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastic RSI (Masters-style)
--------------------------------
Implements the code shown on pages 111–112:
  1) Build RSI series with lookback `lookback`.
  2) Compute stochastic (%K) of that RSI over `lookback2`.
  3) Optionally EMA-smooth the StochRSI with effective lookback `n_to_smooth`.

Returns arrays with NaN for bars before there’s enough history:
    front_bad = lookback + lookback2 - 1
"""

from __future__ import annotations
import numpy as np

EPS = 1e-60  # same tiny constant used in the book to avoid divide-by-zero


def rsi_series(close: np.ndarray, lookback: int) -> np.ndarray:
    """
    RSI using the exact recurrence Masters shows:
        - initialize with sums over the first (lookback-1) diffs
        - then update with the weighted recurrence (no extra smoothing tricks)
    """
    close = np.asarray(close, dtype=float)
    n = close.size
    out = np.full(n, np.nan)
    if lookback < 2 or n < lookback:
        return out

    # --- initialize over the first (lookback-1) diffs
    upsum = dnsum = EPS
    for i in range(1, lookback):
        diff = close[i] - close[i - 1]
        if diff > 0.0:
            upsum += diff
        else:
            dnsum -= diff  # add magnitude

    upsum /= (lookback - 1.0)
    dnsum /= (lookback - 1.0)

    # --- main loop
    for i in range(lookback, n):
        diff = close[i] - close[i - 1]
        if diff > 0.0:
            upsum = ((lookback - 1.0) * upsum + diff) / lookback
            dnsum *= (lookback - 1.0) / lookback
        else:
            dnsum = ((lookback - 1.0) * dnsum - diff) / lookback
            upsum *= (lookback - 1.0) / lookback

        out[i] = 100.0 * upsum / (upsum + dnsum + EPS)

    return out


def stoch_rsi(
    close: np.ndarray,
    lookback: int,
    lookback2: int,
    n_to_smooth: int = 0,
) -> np.ndarray:
    """
    Stochastic RSI:
      - build RSI with window `lookback`
      - compute %K over RSI with window `lookback2`
      - if n_to_smooth > 1, EMA-smooth with alpha = 2/(n_to_smooth+1)

    Returns array of length len(close) with NaNs before front_bad.
    """
    close = np.asarray(close, dtype=float)
    n = close.size
    out = np.full(n, np.nan)

    if lookback < 2 or lookback2 < 1 or n < lookback + lookback2 - 1:
        return out

    rsi = rsi_series(close, lookback)
    front_bad = lookback + lookback2 - 1

    # --- raw StochRSI (%K of the RSI)
    for i in range(front_bad, n):
        # window over RSI: [i-lookback2+1, i]
        window = rsi[i - lookback2 + 1 : i + 1]
        min_val = np.min(window)
        max_val = np.max(window)
        out[i] = 100.0 * (rsi[i] - min_val) / (max_val - min_val + EPS)

    # --- optional EMA smoothing
    if n_to_smooth and n_to_smooth > 1:
        alpha = 2.0 / (n_to_smooth + 1.0)
        sm = out[front_bad]
        for i in range(front_bad, n):
            sm = alpha * out[i] + (1.0 - alpha) * sm
            out[i] = sm

    return out


# ---------------- Demo / quick test ----------------
if __name__ == "__main__":
    # Tiny demo with synthetic closes (replace with your data)
    c = np.array([100, 102, 101, 103, 104, 100, 98, 99, 101, 102, 101, 103], dtype=float)

    srsi_raw = stoch_rsi(c, lookback=5, lookback2=5, n_to_smooth=0)
    srsi_s10 = stoch_rsi(c, lookback=5, lookback2=5, n_to_smooth=10)  # smoothed

    np.set_printoptions(precision=4, suppress=True)
    print("StochRSI (raw):     ", srsi_raw)
    print("StochRSI (EMA L=10):", srsi_s10)
