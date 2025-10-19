# smoke_test.py  (run from the project root:  python3 smoke_test.py)
import numpy as np
from test_indicators import (
    compute_msi, index_returns, cma_oos_signal,
    rs_panel, rs_fractiles, offense_index, defense_index,
    rss, drss,
    dom_index, doe_index,
    relative_momentum, top_n_signal,
)

rng = np.random.default_rng(7)
n_markets, n = 6, 300

# toy prices that wander upward with noise
close = np.cumsum(0.02 + 0.8*rng.standard_normal((n_markets, n)), axis=1) + 100
close = np.maximum(close, 1e-6)

# --- Market Sentiment Index & returns ---
msi = compute_msi(close, lookback=60)                 # (n,)
ret = index_returns(close, horizon=5)                 # (n,)
sig = cma_oos_signal(close[0], lookback=20, win=40)   # (n,)

# --- Offense/Defense + RS family ---
rs = rs_panel(close, lookback=60)                     # (M, n)
q20, q50, q80 = rs_fractiles(rs[:, -120:], (0.2, 0.5, 0.8))
off = offense_index(rs)                               # (n,)
defn = defense_index(rs)                              # (n,)

# --- Trend feedback / spread ---
rss_series = rss(close, lookback=60)                  # (n,)
drss_series = drss(rss_series, k=10)                  # (n,)

# --- Direction of momentum / entropy ---
dom = dom_index(close, lookback=10)                   # (n,)
doe = doe_index(close, lookback=10)                   # (n,)

# --- Relative momentum demo ---
rm = relative_momentum(close, lookback=60)            # (M, n)
long_sig = top_n_signal(rm, top_n=2)                  # (M, n) one-hot per column

print("MSI (tail):", np.round(msi[-5:], 3))
print("Index returns (tail):", np.round(ret[-5:], 3))
print("CMA-OOS signal (tail):", np.round(sig[-5:], 3))
print("Offense/Defense (tail):", np.round(off[-5:], 3), np.round(defn[-5:], 3))
print("RSS/ΔRSS (tail):", np.round(rss_series[-5:], 3), np.round(drss_series[-5:], 3))
print("DOM/DOE (tail):", np.round(dom[-5:], 3), np.round(doe[-5:], 3))
print("Top-N long breadth (tail):", long_sig[:, -1].sum())
