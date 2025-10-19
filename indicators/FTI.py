#!/usr/bin/env python3
# fti.py — Khalsa/Masters FTI in Python (faithful port of FTI.CPP)
# - Zero-lag FIR low-pass per Otnes (as in Masters’ code)
# - “Legs” measured between turning points to compute FTI
# - Width from fractile beta of |actual - filtered|
# - Best period/width/FTI selection via local maxima and sorting
#
# No external deps beyond numpy.

from __future__ import annotations
import math
from typing import List
import numpy as np

# ---------- Regularized lower incomplete gamma P(a, x) (Numerical Recipes) ----------
def gammainc_lower_reg(a: float, x: float, tol: float = 3e-14, itmax: int = 200) -> float:
    """Regularized lower incomplete gamma: P(a,x) = γ(a,x)/Γ(a)."""
    if x <= 0.0:
        return 0.0
    if a <= 0.0:
        raise ValueError("a must be > 0 for gammainc")

    # Use series when x < a+1; else use continued fraction for Q and return 1-Q
    if x < a + 1.0:
        ap = a
        summ = 1.0 / a
        delt = summ
        for _ in range(itmax):
            ap += 1.0
            delt *= x / ap
            summ += delt
            if abs(delt) < abs(summ) * tol:
                break
        return summ * math.exp(-x + a * math.log(x) - math.lgamma(a))
    else:
        # Continued fraction for Q(a,x)
        b = x + 1.0 - a
        c = 1.0 / 1e-30
        d = 1.0 / b
        h = d
        for i in range(1, itmax + 1):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < tol:
                break
        q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
        return 1.0 - q


class FTI:
    """
    Python port of the FTI class from Masters’ FTI.CPP.
    Constructor args match the C++:
      use_log     : int (0/1) — if 1, operate on log(price)
      min_period  : shortest period (>=2)
      max_period  : longest period
      half_length : half the FIR length (total taps = 2*half_length+1); must satisfy 2*half_length >= max_period
      lookback    : processing block length; channel length = lookback - half_length
      beta        : fractile (e.g., 0.95) for channel width
      noise_cut   : fraction of longest leg to define noise (e.g., 0.2)
    After each .process(close, icase), access:
      get_filtered_value(period), get_width(period), get_fti(period), get_sorted_index(which)
    """
    def __init__(self,
                 use_log: int,
                 min_period: int,
                 max_period: int,
                 half_length: int,
                 lookback: int,
                 beta: float,
                 noise_cut: float):
        self.error = 0
        self.use_log = int(use_log)
        self.min_period = int(min_period)
        self.max_period = int(max_period)
        self.half_length = int(half_length)
        self.lookback = int(lookback)
        self.beta = float(beta)
        self.noise_cut = float(noise_cut)

        nper = self.max_period - self.min_period + 1
        self.y = np.empty(self.lookback + self.half_length, dtype=float)
        self.coefs = np.empty((nper, self.half_length + 1), dtype=float)
        self.filtered = np.zeros(nper, dtype=float)
        self.width = np.zeros(nper, dtype=float)
        self.fti = np.zeros(nper, dtype=float)
        self.sorted_idx = np.zeros(nper, dtype=int)

        # Precompute FIR coefs per period
        for p in range(self.min_period, self.max_period + 1):
            self.coefs[p - self.min_period, :] = self._find_coefs(p)

    # ----- FIR coefficient builder (Otnes / as in C++) -----
    def _find_coefs(self, period: int) -> np.ndarray:
        c = np.zeros(self.half_length + 1, dtype=float)
        d = np.array([0.35577019, 0.2436983, 0.07211497, 0.00630165], dtype=float)

        fact = 2.0 / float(period)
        c[0] = fact

        fact *= math.pi
        for i in range(1, self.half_length + 1):
            c[i] = math.sin(i * fact) / (i * math.pi)

        # Taper endpoint
        c[self.half_length] *= 0.5

        # Otnes windowing and normalization
        sumg = c[0]
        for i in range(1, self.half_length + 1):
            sm = float(d[0])
            ang = i * math.pi / self.half_length
            for j in range(1, 4):
                sm += 2.0 * d[j] * math.cos(j * ang)
            c[i] *= sm
            sumg += 2.0 * c[i]

        c /= sumg
        return c

    # ----- main processing of one bar (close[icase] is the "current" bar) -----
    def process(self, close: np.ndarray, icase: int, chronological: bool = True) -> None:
        # 1) build chronological block y[0..lookback-1], with y[-1] = current
        if chronological:
            # take last lookback values ending at icase
            sl = slice(icase - self.lookback + 1, icase + 1)
            if sl.start < 0:
                raise IndexError("Not enough history for the given lookback at icase")
            y_base = close[sl]
        else:
            # emulate reversed pointer (not used in Masters’ driver)
            y_base = close[icase:icase + self.lookback][::-1]

        if self.use_log:
            self.y[:self.lookback] = np.log(np.asarray(y_base, dtype=float))
        else:
            self.y[:self.lookback] = np.asarray(y_base, dtype=float)

        # 2) fit least-squares line to last (half_length+1) points and extend half_length forward
        xmean = -0.5 * self.half_length
        ytail = self.y[self.lookback - 1 - np.arange(self.half_length + 1)]
        ymean = float(ytail.mean())

        xsq = 0.0
        xy = 0.0
        for i in range(self.half_length + 1):
            xdiff = -i - xmean
            ydiff = ytail[i] - ymean
            xsq += xdiff * xdiff
            xy += xdiff * ydiff
        slope = xy / (xsq if xsq != 0.0 else 1e-30)

        for i in range(self.half_length):
            self.y[self.lookback + i] = (i + 1.0 - xmean) * slope + ymean

        # 3) for each period, convolve FIR over channel and compute width/legs/fti
        nper = self.max_period - self.min_period + 1
        ch_start = self.half_length
        ch_end = self.lookback - 1
        ch_len = ch_end - ch_start + 1  # == lookback - half_length

        diff_work = np.empty(ch_len, dtype=float)
        leg_work = np.empty(self.lookback, dtype=float)  # upper bound on legs

        for p in range(self.min_period, self.max_period + 1):
            co = self.coefs[p - self.min_period]

            # Reset leg tracking
            extreme_type = 0
            extreme_value = 0.0
            n_legs = 0
            longest_leg = 0.0
            prior = 0.0

            # Channel convolution
            for iy in range(ch_start, ch_end + 1):
                s = co[0] * self.y[iy]
                # symmetric taps
                for k in range(1, self.half_length + 1):
                    s += co[k] * (self.y[iy + k] + self.y[iy - k])

                if iy == ch_end:
                    self.filtered[p - self.min_period] = s

                diff_work[iy - ch_start] = abs(self.y[iy] - s)

                # legs bookkeeping (turning points of filtered signal)
                if iy == ch_start:
                    extreme_type = 0
                    extreme_value = s
                    n_legs = 0
                    longest_leg = 0.0

                elif extreme_type == 0:
                    if s > extreme_value:
                        extreme_type = -1  # first point was a low
                    elif s < extreme_value:
                        extreme_type = 1   # first point was a high

                elif iy == ch_end:
                    if extreme_type:
                        leg_work[n_legs] = abs(extreme_value - s)
                        n_legs += 1
                        if leg_work[n_legs - 1] > longest_leg:
                            longest_leg = leg_work[n_legs - 1]

                else:
                    if extreme_type == 1 and s > prior:
                        # turned up: leg from last high to recent low (prior)
                        leg_work[n_legs] = extreme_value - prior
                        n_legs += 1
                        if leg_work[n_legs - 1] > longest_leg:
                            longest_leg = leg_work[n_legs - 1]
                        extreme_type = -1
                        extreme_value = prior
                    elif extreme_type == -1 and s < prior:
                        # turned down: leg from last low to recent high (prior)
                        leg_work[n_legs] = prior - extreme_value
                        n_legs += 1
                        if leg_work[n_legs - 1] > longest_leg:
                            longest_leg = leg_work[n_legs - 1]
                        extreme_type = 1
                        extreme_value = prior

                prior = s

            # Width = beta fractile of |actual - filtered|
            # Index in C++: i = int(beta*(lookback - half_length + 1)) - 1
            n_diff = ch_len
            idx = int(self.beta * (self.lookback - self.half_length + 1)) - 1
            if idx < 0:
                idx = 0
            if idx > n_diff - 1:
                idx = n_diff - 1
            w = float(np.sort(diff_work)[idx])
            self.width[p - self.min_period] = w

            # Mean of legs above noise threshold
            if n_legs == 0:
                self.fti[p - self.min_period] = 0.0
            else:
                noise_lvl = self.noise_cut * longest_leg
                legs = leg_work[:n_legs]
                mask = legs > noise_lvl
                if not np.any(mask):
                    self.fti[p - self.min_period] = 0.0
                else:
                    mean_move = float(legs[mask].mean())
                    self.fti[p - self.min_period] = mean_move / (w + 1e-5)

        # 4) sort local maxima of FTI (including endpoints) descending
        candidates: List[int] = []
        for i in range(nper):
            if i == 0 or i == nper - 1 or (self.fti[i] >= self.fti[i - 1] and self.fti[i] >= self.fti[i + 1]):
                candidates.append(i)
        # sort by descending FTI
        order = sorted(candidates, key=lambda i: -self.fti[i])
        # fill sorted_idx (remaining entries left as 0)
        self.sorted_idx[:len(order)] = order
        if len(order) < nper:
            self.sorted_idx[len(order):] = 0

    # --- accessors ---
    def get_filtered_value(self, period: int) -> float:
        return float(self.filtered[period - self.min_period])

    def get_width(self, period: int) -> float:
        return float(self.width[period - self.min_period])

    def get_fti(self, period: int) -> float:
        return float(self.fti[period - self.min_period])

    def get_sorted_index(self, which: int) -> int:
        idx = int(self.sorted_idx[which])
        if idx < 0 or idx > self.max_period - self.min_period:
            raise IndexError("sorted index out of range")
        return idx


# ---------- Convenience wrappers (mirror COMP_VAR logic) ----------
def fti_lowpass_series(close: np.ndarray,
                       lookback: int,
                       half_length: int,
                       min_period: int,
                       max_period: int,
                       use_log: int = 1,
                       beta: float = 0.95,
                       noise_cut: float = 0.20) -> np.ndarray:
    n = len(close)
    out = np.zeros(n, dtype=float)
    front_bad = lookback - 1
    if n <= front_bad:
        return out
    fti = FTI(use_log, min_period, max_period, half_length, lookback, beta, noise_cut)
    for icase in range(front_bad, n):
        fti.process(np.asarray(close, dtype=float), icase, True)
        k = fti.get_sorted_index(0)  # index into [min_period..max_period]
        period = min_period + k
        val = fti.get_filtered_value(period)
        out[icase] = math.exp(val) if use_log else val
    return out


def fti_best_period_series(close: np.ndarray, lookback: int, half_length: int,
                           min_period: int, max_period: int,
                           use_log: int = 1, beta: float = 0.95, noise_cut: float = 0.20) -> np.ndarray:
    n = len(close)
    out = np.zeros(n, dtype=float)
    front_bad = lookback - 1
    if n <= front_bad:
        return out
    f = FTI(use_log, min_period, max_period, half_length, lookback, beta, noise_cut)
    for icase in range(front_bad, n):
        f.process(np.asarray(close, dtype=float), icase, True)
        out[icase] = min_period + f.get_sorted_index(0)
    return out


def fti_best_width_series(close: np.ndarray, lookback: int, half_length: int,
                          min_period: int, max_period: int,
                          use_log: int = 1, beta: float = 0.95, noise_cut: float = 0.20) -> np.ndarray:
    n = len(close)
    out = np.zeros(n, dtype=float)
    front_bad = lookback - 1
    if n <= front_bad:
        return out
    f = FTI(use_log, min_period, max_period, half_length, lookback, beta, noise_cut)
    for icase in range(front_bad, n):
        f.process(np.asarray(close, dtype=float), icase, True)
        k = f.get_sorted_index(0)
        period = min_period + k
        if use_log:
            value = f.get_filtered_value(period)
            term = f.get_width(period)
            out[icase] = 0.5 * (math.exp(value + term) - math.exp(value - term))
        else:
            out[icase] = f.get_width(period)
    return out


def fti_best_fti_series(close: np.ndarray, lookback: int, half_length: int,
                        min_period: int, max_period: int,
                        use_log: int = 1, beta: float = 0.95, noise_cut: float = 0.20) -> np.ndarray:
    """
    Mirrors: output = 100 * igamma(2.0, FTI/3.0) - 50.0
    where igamma is the (lower) regularized incomplete gamma.
    """
    n = len(close)
    out = np.zeros(n, dtype=float)
    front_bad = lookback - 1
    if n <= front_bad:
        return out
    f = FTI(use_log, min_period, max_period, half_length, lookback, beta, noise_cut)
    for icase in range(front_bad, n):
        f.process(np.asarray(close, dtype=float), icase, True)
        k = f.get_sorted_index(0)
        period = min_period + k
        raw = f.get_fti(period)
        out[icase] = 100.0 * gammainc_lower_reg(2.0, raw / 3.0) - 50.0
    return out


# --------------- quick demo ---------------
if __name__ == "__main__":
    # Synthetic walk for smoke test
    np.random.seed(0)
    n = 300
    close = np.cumsum(np.random.normal(0, 1, size=n)).astype(float) + 100.0
    close = np.maximum(close, 1e-6)

    lookback = 60
    half_length = 12
    min_period, max_period = 5, 30

    lp = fti_lowpass_series(close, lookback, half_length, min_period, max_period)
    bp = fti_best_period_series(close, lookback, half_length, min_period, max_period)
    bw = fti_best_width_series(close, lookback, half_length, min_period, max_period)
    bf = fti_best_fti_series(close, lookback, half_length, min_period, max_period)

    np.set_printoptions(precision=3, suppress=True)
    print("Low-pass (tail):", lp[-5:])
    print("Best period (tail):", bp[-5:])
    print("Best width (tail):", bw[-5:])
    print("Best FTI (tail):", bf[-5:])
