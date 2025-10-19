# rsi_wilder.py
# Exact port of the book's RSI fragment (Wilder smoothing).
# - First valid RSI appears at index == lookback (earlier values are None)
# - Uses tiny epsilon to avoid divide-by-zero (1e-60 in the original)

from typing import List, Optional

def rsi_wilder(close: List[float], lookback: int) -> List[Optional[float]]:
    """
    Compute classic RSI with Wilder's smoothing exactly like the C snippet:
      - Initialize avg up/down over the first (lookback-1) diffs
      - Then update with the recursive Wilder averages
      - Output RSI = 100 * up / (up + down)

    Parameters
    ----------
    close : list of close prices (chronological order)
    lookback : smoothing length (e.g., 14)

    Returns
    -------
    List[Optional[float]] : RSI values; indices < lookback are None
    """
    n = len(close)
    if lookback < 2 or n < lookback + 1:
        # Not enough data to match the book's indexing (first RSI at index lookback)
        return [None] * n

    out: List[Optional[float]] = [None] * n
    eps = 1e-60

    # --- Initialization (matches the book) ---
    # front_bad = lookback  // first valid RSI at this index
    upsum = eps
    dnsum = eps
    for icase in range(1, lookback):
        diff = close[icase] - close[icase - 1]
        if diff > 0.0:
            upsum += diff
        else:
            dnsum -= diff  # add magnitude of down move

    upsum /= (lookback - 1.0)
    dnsum /= (lookback - 1.0)

    # --- Main loop (start computing) ---
    for icase in range(lookback, n):
        diff = close[icase] - close[icase - 1]
        if diff > 0.0:
            # up gets new diff; down decays (implicitly adds 0)
            upsum  = ((lookback - 1.0) * upsum + diff) / lookback
            dnsum  = ((lookback - 1.0) * dnsum         ) / lookback
        else:
            # down gets |diff|; up decays
            dnsum  = ((lookback - 1.0) * dnsum + (-diff)) / lookback
            upsum  = ((lookback - 1.0) * upsum          ) / lookback

        out[icase] = 100.0 * upsum / (upsum + dnsum)

    return out


def rsi_centered(close: List[float], lookback: int) -> List[Optional[float]]:
    """
    Book's suggestion for modeling: center RSI at 0 by subtracting 50.
    """
    r = rsi_wilder(close, lookback)
    return [None if v is None else (v - 50.0) for v in r]


# ---- quick example ----
if __name__ == "__main__":
    prices = [100, 101, 102, 101, 103, 103, 102, 104, 105, 104, 106, 107, 106, 108, 109]
    rsi14 = rsi_wilder(prices, lookback=14)
    rsi14c = rsi_centered(prices, lookback=14)
    for i, (p, r, c) in enumerate(zip(prices, rsi14, rsi14c)):
        print(f"{i:2d}  close={p:6.2f}  RSI={str(None if r is None else round(r,2)):>6}  RSI-50={str(None if c is None else round(c,2)):>6}")
