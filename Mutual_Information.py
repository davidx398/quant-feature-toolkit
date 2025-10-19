#!/usr/bin/env python3
"""
Masters — MUTUAL INFORMATION indicator (port of VAR_MUTUAL_INFORMATION)

Window size:
    needed = (2^(word_length+1)) * mult + 1    # '+1' because we work on differences
    front_bad = needed - 1                      # indices < front_bad are neutral 0

Binary symbolization:
    diff > 0 -> 1, else 0 (ties count as 0, matching his style)

Mutual information in natural logs:
    MI = sum_{x in 2^word, y in {0,1}} p(x,y) * ln( p(x,y) / (p(x)*p(y)) )

Post-processing (Masters):
    value = MI * mult * sqrt(word_length) - 0.12*word_length - 0.04
    output = 100 * Phi(3 * value) - 50
"""

from __future__ import annotations
import math
from typing import Iterable, List

EPS = 1e-300  # tiny guard for logs/divisions


def normal_cdf(z: float) -> float:
    """Standard normal CDF via erf (no NumPy required)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _mi_on_window(prices: List[float], word_len: int) -> float:
    """
    Mutual information between:
      X = previous `word_len` bits (2^word_len states),
      Y = next bit (2 states),
    computed over a single rolling window of prices.
    """
    m = len(prices)
    if word_len < 1 or m < word_len + 2:
        return 0.0

    # 1) diffs -> bits (ties -> 0)
    bits = [1 if prices[i] > prices[i - 1] else 0 for i in range(1, m)]
    L = len(bits)  # = m - 1
    if L < word_len + 1:
        return 0.0

    # 2) count joint occurrences of (X word, Y next bit)
    nX = 1 << word_len
    mask = nX - 1
    joint = [[0, 0] for _ in range(nX)]  # joint[x][y]

    # initial word for t = word_len-1
    w = 0
    for i in range(word_len):
        w = (w << 1) | bits[i]

    # for t from word_len-1 .. L-2:
    #   X = bits[t-word_len+1..t] (encoded in w)
    #   Y = bits[t+1]
    for t in range(word_len - 1, L - 1):
        y = bits[t + 1]
        joint[w][y] += 1
        # roll word to include next bit for next t
        w = ((w << 1) & mask) | bits[t + 1]

    total = sum(joint_x[0] + joint_x[1] for joint_x in joint)
    if total <= 0:
        return 0.0

    # 3) mutual information in nats
    py0 = sum(joint_x[0] for joint_x in joint) / total
    py1 = 1.0 - py0
    mi = 0.0
    for x in range(nX):
        px = (joint[x][0] + joint[x][1]) / total
        if px <= 0:  # skip empty X states
            continue
        for y, py in ((0, py0), (1, py1)):
            pxy = joint[x][y] / total
            if pxy > 0 and py > 0:
                mi += pxy * math.log(pxy / (px * py + EPS) + EPS)
    return max(mi, 0.0)  # tiny negative from round-off → 0


def mutual_information_indicator(
    close: Iterable[float],
    word_length: int,
    mult: int,
) -> List[float]:
    """
    Port of the C++ VAR_MUTUAL_INFORMATION branch.

    Parameters
    ----------
    close        : price series (list/iterable)
    word_length  : number of past bits in X (>=1)
    mult         : samples-per-bin multiplier (>=1)

    Returns
    -------
    List[float] : indicator values, with leading neutral zeros up to front_bad-1.
    """
    c = list(map(float, close))
    n = len(c)
    if n == 0:
        return []

    if word_length < 1:
        raise ValueError("word_length must be >= 1")
    if mult < 1:
        mult = 1

    # needed = 2^(word_length+1) * mult + 1  (the +1 is for differences)
    needed = (1 << (word_length + 1)) * mult + 1
    front_bad = min(max(0, needed - 1), n)

    out = [0.0] * n  # neutral leading region

    for icase in range(front_bad, n):
        start = icase - (needed - 1)
        window = c[start : icase + 1]  # chronological order

        mi = _mi_on_window(window, word_length)

        # Masters' scaling & compression
        value = mi * mult * math.sqrt(float(word_length)) - 0.12 * word_length - 0.04
        out[icase] = 100.0 * normal_cdf(3.0 * value) - 50.0

    return out


# ---------------- demo ----------------
if __name__ == "__main__":
    # Simple synthetic series: gentle uptrend then chop
    import random
    random.seed(2)
    trend = [100 + 0.15 * i for i in range(180)]
    chop  = [trend[-1] + (random.random() - 0.5) * 0.6 for _ in range(180)]
    close = trend + chop

    vals = mutual_information_indicator(close, word_length=3, mult=4)
    print([round(v, 2) for v in vals[-30:]])
