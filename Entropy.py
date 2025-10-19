#!/usr/bin/env python3
"""
Masters — ENTROPY indicator (port of the VAR_ENTROPY branch)

Idea
----
- Convert a rolling window of prices to a binary sequence of *differences*:
  diff > 0 → 1, else → 0 (ties count as 0 like in his code style).
- Form overlapping binary “words” of length `word_length` (there are 2^word_length bins).
- Compute normalized Shannon entropy of those word frequencies.
- Apply Masters’ transforms and final compression:
    if word_length == 1:
        value = 1 - exp( log(1.00000001 - H) / 5 )
        mean  = 0.6
    else:
        value = 1 - exp( log(1 - H) / word_length )
        mean  = 1/word_length + 0.35
    output = 100 * Phi( 8 * (value - mean) ) - 50

Window size
-----------
needed = 2**word_length * mult  # "mult per bin" samples
needed += 1                      # +1 because entropy works on *differences*
front_bad = needed - 1           # first valid index (0..front_bad-1 are neutral 0.0)

Returns
-------
List[float] of length n, with leading neutral zeros up to front_bad-1.
"""

from __future__ import annotations
import math
from typing import Iterable, List
import sys
import pandas as pd
EPS = 1e-30  # tiny guard for logs/divisions


def normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erf (no NumPy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _binary_word_entropy_from_prices(prices: List[float], word_len: int) -> float:
    """
    Compute normalized Shannon entropy H in [0, 1] from a *single* window of prices.

    Steps:
      1) diffs -> bits (1 if up, else 0)
      2) sliding binary words of length `word_len`
      3) histogram over the 2**word_len bins
      4) H = -sum p*ln p / (word_len * ln 2)
    """
    if word_len < 1:
        return 0.0
    m = len(prices)
    if m < word_len + 1:
        return 0.0

    # 1) Differences to bits (ties -> 0)
    bits = [1 if prices[i] > prices[i - 1] else 0 for i in range(1, m)]
    L = len(bits)
    if L < word_len:
        return 0.0

    # 2) Sliding word counts via bit rolling
    n_bins = 1 << word_len
    counts = [0] * n_bins
    mask = n_bins - 1

    # first word
    w = 0
    for i in range(word_len):
        w = (w << 1) | bits[i]
    counts[w] += 1

    # remaining words
    for i in range(word_len, L):
        w = ((w << 1) & mask) | bits[i]
        counts[w] += 1

    total = sum(counts)
    if total <= 0:
        return 0.0

    # 3) Shannon entropy (natural log), normalized to [0, 1]
    H = 0.0
    for c in counts:
        if c:
            p = c / total
            H -= p * math.log(p + EPS)
    H /= (word_len * math.log(2.0))  # normalize by log(2^word_len)
    # numeric guard
    if H < 0.0: H = 0.0
    if H > 1.0: H = 1.0
    return H


def entropy_indicator(close: Iterable[float], word_length: int, mult: int) -> List[float]:
    """
    Port of the C++ 'VAR_ENTROPY' branch.

    Parameters
    ----------
    close        : sequence of prices
    word_length  : length of binary words (>=1)
    mult         : samples-per-bin multiplier (>=1); window uses 2^word_length * mult + 1 prices

    Returns
    -------
    list[float] : ENTROPY indicator ([-50, 50] style scale), with leading neutral zeros
    """
    c = list(map(float, close))
    n = len(c)
    if n == 0:
        return []

    if word_length < 1:
        raise ValueError("word_length must be >= 1")
    if mult < 1:
        mult = 1

    needed = (1 << word_length) * mult
    needed += 1  # "Plus one because computing differences"
    front_bad = max(0, needed - 1)
    front_bad = min(front_bad, n)

    out = [0.0] * n  # Set undefined/early region to neutral

    # main loop
    for icase in range(front_bad, n):
        start = icase - (needed - 1)
        window = c[start:icase + 1]  # chronological order

        # Raw normalized entropy in [0, 1]
        H = _binary_word_entropy_from_prices(window, word_length)

        # Masters’ transforms & compression
        if word_length == 1:
            # avoid log(0) with tiny 1.00000001 fudge like the C++
            value = 1.0 - math.exp(math.log(1.00000001 - H) / 5.0)
            mean = 0.6
        else:
            value = 1.0 - math.exp(math.log(1.0 - H + EPS) / float(word_length))
            mean = 1.0 / float(word_length) + 0.35

        out[icase] = 100.0 * normal_cdf(8.0 * (value - mean)) - 50.0

    return out

def compute_entropy_csv(csv_path: str, column: str = "CMMA", word: int = 3, mult: int = 4, out_path: str | None = None):
    df = pd.read_csv(csv_path)
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)

    if column not in df.columns:
        raise SystemExit(f"Column '{column}' not found. Available: {list(df.columns)}")

    ent_col = f"ENTROPY_{column}_w{word}_m{mult}"
    df[ent_col] = entropy_indicator(df[column].astype(float).tolist(), word, mult)

    if out_path is None:
        stem, ext = (csv_path.rsplit(".", 1) + ["csv"])[:2]
        out_path = f"{stem}_entropy_{column}_w{word}_m{mult}.{ext}"
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 entropy_from_csv.py /path/to/file.csv [column] [word] [mult]")
        print("Defaults: column=CMMA, word=3, mult=4")
        sys.exit(1)
    csv = sys.argv[1]
    col  = sys.argv[2] if len(sys.argv) > 2 else "CMMA"
    word = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    mult = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    compute_entropy_csv(csv, column=col, word=word, mult=mult)
# ------------------ demo ------------------
# ------------------ CLI to compute ENTROPY(CMMA) from a CSV ------------------
