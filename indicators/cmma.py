# cmma_numpy.py
import numpy as np

EPS = 1e-60

def _normal_cdf_vec(x: np.ndarray) -> np.ndarray:
    # erf-free approximation (Abramowitz–Stegun), avoids SciPy/np.erf issues
    x = np.asarray(x, float)
    sign = np.sign(x)
    ax = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    erf_approx = sign * (1.0 - y * np.exp(-ax * ax))
    return 0.5 * (1.0 + erf_approx)

def _atr_log_window(icase: int, length: int,
                    h: np.ndarray, l: np.ndarray, c: np.ndarray) -> float:
    """Average 'true range' in log domain over [icase-length+1 .. icase]."""
    if length == 0:
        return float(np.log(h[icase] / max(l[icase], EPS)))
    start = icase - length + 1
    hw = h[start:icase+1]
    lw = l[start:icase+1]
    cprev = c[start-1:icase]  # aligned
    term = np.maximum.reduce([
        hw / np.maximum(lw, EPS),
        hw / np.maximum(cprev, EPS),
        np.maximum(cprev, EPS) / np.maximum(lw, EPS),
    ])
    return float(np.log(term).sum() / length)

def cmma_np(close, high, low, open_, lookback: int, atr_length: int, compress: float = 1.0) -> np.ndarray:
    c = np.asarray(close, float)
    h = np.asarray(high,  float)
    l = np.asarray(low,   float)
    o = np.asarray(open_, float)  # not used by ATR(log); kept for signature parity
    n = c.size
    if not (h.size == l.size == o.size == n):
        raise ValueError("All input series must have the same length.")
    if lookback < 1 or atr_length < 0:
        raise ValueError("lookback >= 1 and atr_length >= 0 required.")

    out = np.zeros(n, float)
    if n == 0:
        return out

    front_bad = max(lookback, atr_length)
    if n <= front_bad:
        return out

    logc = np.log(np.maximum(c, EPS))

    # Rolling mean of log close EXCLUDING current bar:
    # ma_log[i] = mean(logc[i-lookback : i])
    # Use cumsum for O(1) updates.
    cs = np.r_[0.0, np.cumsum(logc)]
    # valid only from i >= lookback; we’ll access inside the loop to stay clear
    for i in range(front_bad, n):
        ma_log = (cs[i] - cs[i - lookback]) / lookback  # excludes current bar
        denom = _atr_log_window(i, atr_length, h, l, c)
        if denom <= 0.0:
            out[i] = 0.0
            continue
        denom *= np.sqrt(lookback + 1.0)  # Masters’ extra scaling
        z = (logc[i] - ma_log) / (denom + EPS)
        out[i] = 100.0 * _normal_cdf_vec(compress * z) - 50.0

    # leading region left as 0.0 (book leaves undefined; 0.0 is a harmless placeholder)
    return out

# --- quick smoke test ---
# if __name__ == "__main__":
#     close = np.array([100,101,102,101,103,104,103,105,106,104,107,108,109,108,110], float)
#     high  = close * 1.01
#     low   = close * 0.99
#     open_ = close.copy()
#     vals = cmma_np(close, high, low, open_, lookback=5, atr_length=5, compress=1.0)
#     print("CMMA:", [round(float(x), 4) for x in vals])


