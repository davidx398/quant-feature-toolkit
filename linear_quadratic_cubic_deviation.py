# polynomial_deviation.py
# Linear / Quadratic / Cubic deviation from expectation (Masters, pp. 161–163)
# - Uses log(close)
# - Projects onto normalized discrete Legendre polynomials
# - Normalizes by RMS error over the lookback window
# - Mild compression via normal_cdf(0.6 * z) → [-50, 50] style scale

from __future__ import annotations
import math
from typing import List, Tuple, Iterable

def normal_cdf(x: float) -> float:
    """Standard Normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def legendre_3(n: int) -> Tuple[List[float], List[float], List[float]]:
    """
    Discrete, normalized, *orthogonal* Legendre-like vectors over n points:
      c1 ~ linear, c2 ~ quadratic, c3 ~ cubic
    Matches the C code logic (normalize each to unit length; center c2/c3; 
    remove any tiny projection of c1 from c3).
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    # c1: centered linear ramp in [-1, 1], normalized to unit length
    c1 = [2.0 * i / (n - 1.0) - 1.0 for i in range(n)]
    s = math.sqrt(sum(v*v for v in c1))
    c1 = [v / s for v in c1]

    # c2: square of c1, centered, normalized
    c2 = [v*v for v in c1]
    mean = sum(c2) / n
    c2 = [v - mean for v in c2]
    s = math.sqrt(sum(v*v for v in c2)) or 1.0
    c2 = [v / s for v in c2]

    # c3: cube of c1, centered, normalize, then remove projection on c1, renormalize
    c3 = [v*v*v for v in c1]
    mean = sum(c3) / n
    c3 = [v - mean for v in c3]
    s = math.sqrt(sum(v*v for v in c3)) or 1.0
    c3 = [v / s for v in c3]

    proj = sum(a*b for a, b in zip(c1, c3))  # projection of c3 on c1
    c3 = [v - proj*u for v, u in zip(c3, c1)]
    s = math.sqrt(sum(v*v for v in c3)) or 1.0
    c3 = [v / s for v in c3]

    return c1, c2, c3

def _log(x: float) -> float:
    if x <= 0:
        raise ValueError("Close prices must be positive for log().")
    return math.log(x)

def polynomial_deviation(
    close: Iterable[float],
    lookback: int,
    degree: int = 1,          # 1=linear, 2=quadratic, 3=cubic
    compress_k: float = 0.6,  # Masters uses ~0.6 on this page
    neutral_value: float = 0.0
) -> List[float]:
    """
    Returns series of deviation scores in [-50, 50]-ish scale:
      output[t] = 100 * Phi( compress_k * z_t ) - 50, where
      z_t = (log(close[t]) - predicted[t]) / RMS_error_over_window

    Window is *inclusive* of the current bar: [t-lookback+1 .. t].
    Undefined early values (t < lookback-1) are set to `neutral_value` (Masters uses 0.0).
    Enforces minimal lookback: 3 (linear), 4 (quadratic), 5 (cubic).
    """
    c = list(close)
    n = len(c)
    if degree not in (1, 2, 3):
        raise ValueError("degree must be 1, 2, or 3")

    # Enforce Masters' minimums
    min_lb = {1: 3, 2: 4, 3: 5}[degree]
    if lookback < min_lb:
        lookback = min_lb

    # Precompute Legendre weights for this window length
    w1, w2, w3 = legendre_3(lookback)

    out = [neutral_value] * n
    front_bad = lookback - 1  # number of leading undefined values

    # Main loop over valid indices
    for icase in range(front_bad, n):
        start = icase - lookback + 1
        # 1) Project onto Legendre vectors to get coefficients (c0 is mean)
        c0 = 0.0
        c1 = 0.0
        c2 = 0.0
        c3 = 0.0

        # Linear term (and accumulate mean)
        j = 0
        for k in range(start, icase + 1):
            price = _log(c[k])
            c0 += price
            c1 += price * w1[j]
            j += 1
        c0 /= lookback

        # Quadratic?
        if degree >= 2:
            j = 0
            for k in range(start, icase + 1):
                price = _log(c[k])
                c2 += price * w2[j]
                j += 1

        # Cubic?
        if degree >= 3:
            j = 0
            for k in range(start, icase + 1):
                price = _log(c[k])
                c3 += price * w3[j]
                j += 1

        # 2) Compute RMS error over the window using fitted coefficients
        sumsq = 0.0
        j = 0
        for k in range(start, icase + 1):
            pred = c0 + c1 * w1[j]
            if degree >= 2:
                pred += c2 * w2[j]
            if degree >= 3:
                pred += c3 * w3[j]
            diff = _log(c[k]) - pred
            sumsq += diff * diff
            j += 1
        denom = math.sqrt(sumsq / lookback)

        # 3) Predict current bar and create normalized/compressed score
        if denom > 0.0:
            pred = c0 + c1 * w1[-1]
            if degree >= 2:
                pred += c2 * w2[-1]
            if degree >= 3:
                pred += c3 * w3[-1]
            z = (_log(c[icase]) - pred) / denom
            out[icase] = 100.0 * normal_cdf(compress_k * z) - 50.0
        else:
            out[icase] = neutral_value  # perfect fit or constant prices

    return out

# ------------------------- Demo / quick test -------------------------
if __name__ == "__main__":
    # Tiny synthetic example
    prices = [100, 101, 100, 102, 103, 104, 104, 103, 105, 106, 105, 107, 108]
    for deg in (1, 2, 3):
        vals = polynomial_deviation(prices, lookback=5, degree=deg)
        print(f"degree={deg}: {vals}")
