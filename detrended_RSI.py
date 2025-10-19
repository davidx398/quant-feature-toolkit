# rsi_and_detrended.py
# Implements: Wilder RSI (pg 100), short/long RSI (pg 105–106),
# and Detrended RSI via rolling regression (pg 107).

from __future__ import annotations
import numpy as np

_EPS = 1e-60  # book's tiny constant to avoid divide-by-zero

def rsi_wilder(close: np.ndarray, lookback: int) -> np.ndarray:
    """
    Wilder-style RSI exactly as in the book (pg 100):
      - Initialize with averages over the first (lookback-1) diffs.
      - Then update with Wilder smoothing.
      - Returns NaN until index == lookback.
    """
    c = np.asarray(close, dtype=float)
    n = c.size
    out = np.full(n, np.nan)
    if lookback < 2 or n < lookback:
        return out

    upsum = _EPS
    dnsum = _EPS

    # --- init over first (lookback-1) diffs
    for i in range(1, lookback):
        diff = c[i] - c[i - 1]
        if diff > 0.0:
            upsum += diff
        else:
            dnsum -= diff
    upsum /= (lookback - 1.0)
    dnsum /= (lookback - 1.0)

    # --- rolling Wilder updates
    for i in range(lookback, n):
        diff = c[i] - c[i - 1]
        if diff > 0.0:
            upsum = ((lookback - 1.0) * upsum + diff) / lookback
            dnsum = ((lookback - 1.0) * dnsum) / lookback
        else:
            dnsum = ((lookback - 1.0) * dnsum - diff) / lookback  # note: diff <= 0
            upsum = ((lookback - 1.0) * upsum) / lookback

        out[i] = 100.0 * upsum / (upsum + dnsum)

    return out


def _invlogistic_rsi2(rsi: np.ndarray) -> np.ndarray:
    """
    Page 105 note: for short_length == 2, apply the inverse-logistic transform
    (scaled exactly as shown).
        y = -10 * log( 2/(1 + 0.00999*(2*RSI - 100)) - 1 )
    """
    y = rsi.astype(float).copy()
    mask = np.isfinite(y)
    z = np.empty_like(y)
    z[:] = np.nan
    t = 2.0 / (1.0 + 0.00999 * (2.0 * y[mask] - 100.0)) - 1.0
    # guard: tiny negatives from numeric noise
    t = np.clip(t, 1e-300, None)
    z[mask] = -10.0 * np.log(t)
    return z


def detrended_rsi(
    close: np.ndarray,
    short_length: int,
    long_length: int,
    regression_len: int,
    apply_invlogistic_for_short_len2: bool = True,
) -> dict[str, np.ndarray]:
    """
    Reproduces the three-page sequence:
      1) compute short-term RSI into work1 (pg 105)
         - optional inverse-logistic transform if short_length == 2 (same formula)
      2) compute long-term RSI into work2 (pg 106)
      3) detrend: rolling regression of work1 on work2 over `regression_len`,
         then output = (work1 - mean1) - coef * (work2 - mean2), where
         coef = sum((x-mean2)*(y-mean1)) / (sum((x-mean2)^2) + eps)  (pg 107)

    Returns dict with keys: 'rsi_short', 'rsi_long', 'detrended'
    """
    c = np.asarray(close, dtype=float)
    n = c.size

    rsi_s = rsi_wilder(c, short_length)
    if apply_invlogistic_for_short_len2 and short_length == 2:
        rsi_s = _invlogistic_rsi2(rsi_s)

    rsi_l = rsi_wilder(c, long_length)

    det = np.full(n, np.nan)

    # First usable index must have:
    #   - a valid long RSI value, and
    #   - regression_len values ending at the current bar.
    # Long RSI first valid at index = long_length
    front_bad = long_length + regression_len - 1
    if regression_len < 1 or n <= front_bad:
        return {"rsi_short": rsi_s, "rsi_long": rsi_l, "detrended": det}

    # Rolling regression per page 107
    for icase in range(front_bad, n):
        # means over last `regression_len` values (include current case)
        x_window = rsi_l[icase - regression_len + 1 : icase + 1]
        y_window = rsi_s[icase - regression_len + 1 : icase + 1]
        if not (np.all(np.isfinite(x_window)) and np.all(np.isfinite(y_window))):
            continue

        xmean = float(np.mean(x_window))
        ymean = float(np.mean(y_window))

        xdiff = x_window - xmean
        ydiff = y_window - ymean
        xss = float(np.sum(xdiff * xdiff))
        xy = float(np.sum(xdiff * ydiff))

        coef = xy / (xss + _EPS)

        # actual minus predicted, current case only
        xdiff_now = rsi_l[icase] - xmean
        ydiff_now = rsi_s[icase] - ymean
        det[icase] = ydiff_now - coef * xdiff_now

    return {"rsi_short": rsi_s, "rsi_long": rsi_l, "detrended": det}


# --------------------------
# Quick usage example
# --------------------------
if __name__ == "__main__":
    # Dummy close series for a smoke test
    rng = np.random.default_rng(0)
    close = np.cumsum(rng.normal(0, 1, 500)) + 100.0

    out = detrended_rsi(close, short_length=2, long_length=14, regression_len=30)

    print("Last 5 RSI(short):   ", np.round(out["rsi_short"][-5:], 3))
    print("Last 5 RSI(long):    ", np.round(out["rsi_long"][-5:], 3))
    print("Last 5 DetrendedRSI: ", np.round(out["detrended"][-5:], 3))
