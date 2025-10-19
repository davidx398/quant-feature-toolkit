# janus/core.py
from __future__ import annotations
import numpy as np
from .utils import safe_log, median_axis0, argsort, EPS

class JanusEngine:
    """
    NumPy port of Timothy Masters' JANUS class.
    Shapes:
      prices: (M, B)  (markets × bars)
      returns: (M, B-1)  where returns[:, ib] = log(p[:, ib+1]/p[:, ib])
    """

    def __init__(self, prices: np.ndarray, lookback: int,
                 spread_tail: float, min_CMA: int, max_CMA: int):
        if prices.ndim != 2:
            raise ValueError("prices must be 2D (n_markets, nbars)")
        self.prices = prices.astype(float, copy=True)
        self.M, self.B = self.prices.shape
        self.N = self.B - 1                     # n_returns
        self.lookback = int(lookback)
        self.spread_tail = float(spread_tail)
        self.min_CMA = int(min_CMA)
        self.max_CMA = int(max_CMA)

        # allocate all state
        self.returns = np.empty((self.M, self.N), float)
        self.mkt_index_returns = np.empty(self.N, float)
        self.dom_index_returns = np.empty(self.N, float)

        self.rs = np.zeros((self.N, self.M), float)
        self.rs_fractile = np.zeros((self.N, self.M), float)
        self.rs_lagged = np.zeros((self.N, self.M), float)
        self.rs_lookahead = 0
        self.rs_leader = np.zeros(self.N, float)
        self.rs_laggard = np.zeros(self.N, float)
        self.oos_avg = np.zeros(self.N, float)

        self.rm = np.zeros((self.N, self.M), float)
        self.rm_fractile = np.zeros((self.N, self.M), float)
        self.rm_lagged = np.zeros((self.N, self.M), float)
        self.rm_lookahead = 0
        self.rm_leader = np.zeros(self.N, float)
        self.rm_laggard = np.zeros(self.N, float)

        self.rss = np.zeros(self.N, float)
        self.rss_change = np.zeros(self.N, float)
        self.dom = np.zeros((self.N, self.M), float)
        self.doe = np.zeros((self.N, self.M), float)
        self.dom_index = np.zeros(self.N, float)
        self.doe_index = np.zeros(self.N, float)

        # CMA buffers
        rng = self.max_CMA - self.min_CMA + 1
        self.CMA_alpha = np.zeros(rng, float)
        self.CMA_smoothed = np.zeros(rng, float)
        self.CMA_equity = np.zeros(rng, float)
        self.CMA_OOS = np.zeros(self.N, float)
        self.CMA_leader_OOS = np.zeros(self.N, float)

        self._prepare()

    # ---------------- core steps (direct ports) ----------------

    def _prepare(self) -> None:
        # returns per market
        self.returns[:] = safe_log(self.prices[:, 1:]) - safe_log(self.prices[:, :-1])
        # per-bar median across markets == index returns
        self.mkt_index_returns[:] = median_axis0(self.returns)

    def compute_rs(self, lag: int = 0) -> None:
        """Relative Strength (RS) and RS fractiles (port of compute_rs)."""
        L = self.lookback
        self.rs_lookahead = int(lag)

        for ibar in range(L - 1, self.N):
            # reverse-chronological window of index
            idx = self.mkt_index_returns[ibar : ibar - L : -1]  # length L

            # median ignoring first 'lag' slots of idx (they are newest)
            w = idx[lag:]
            med = np.median(w)

            # total offensive/defensive for index
            idx_off = 1e-30
            idx_def = -1e-30
            for val in w:
                d = val - med
                if d >= 0:
                    idx_off += d
                else:
                    idx_def += d

            # per-market RS
            rs_row = np.empty(self.M, float)
            for m in range(self.M):
                off = 0.0
                de  = 0.0
                # map reverse-chrono index[i] to returns[m, ibar - i]
                for i in range(lag, L):
                    d = self.returns[m, ibar - i] - med
                    if idx[i] >= med:
                        off += d
                    else:
                        de  += d
                val = 70.710678 * (off / idx_off - de / idx_def)  # 100/sqrt(2)
                val = min(200.0, max(-200.0, val))
                rs_row[m] = val

            if lag == 0:
                self.rs[ibar] = rs_row
            else:
                self.rs_lagged[ibar] = rs_row

            # fractiles (ascending)
            order = np.argsort(rs_row)
            if lag == 0:
                self.rs_fractile[ibar, order] = np.arange(self.M, dtype=float) / (self.M - 1.0)

    def compute_rss(self) -> None:
        """Relative Strength Spread (port of compute_rss)."""
        L = self.lookback
        for ibar in range(L - 1, self.N):
            srt = np.sort(self.rs[ibar])
            k = int(self.spread_tail * (self.M + 1)) - 1
            k = max(0, k)
            n = k + 1
            width = 0.0
            for j in range(n):
                width += srt[self.M - 1 - j] - srt[j]
            self.rss[ibar] = width / n
            self.rss_change[ibar] = 0.0 if ibar == (L - 1) else self.rss[ibar] - self.rss[ibar - 1]

    def compute_dom_doe(self) -> None:
        """DOM/DOE (port of compute_dom_doe)."""
        L = self.lookback
        dom_sum = np.zeros(self.M, float)
        doe_sum = np.zeros(self.M, float)
        dom_idx_sum = 0.0
        doe_idx_sum = 0.0

        for ibar in range(L - 1, self.N):
            if self.rss_change[ibar] > 0:
                dom_idx_sum += self.mkt_index_returns[ibar]
                dom_sum += self.returns[:, ibar]
            elif self.rss_change[ibar] < 0:
                doe_idx_sum += self.mkt_index_returns[ibar]
                doe_sum += self.returns[:, ibar]
            # write current cum-sums
            self.dom[ibar] = dom_sum
            self.doe[ibar] = doe_sum
            self.dom_index[ibar] = dom_idx_sum
            self.doe_index[ibar] = doe_idx_sum

    def compute_rm(self, lag: int = 0) -> None:
        """Relative Momentum (RM) and fractiles (port of compute_rm)."""
        L = self.lookback
        self.rm_lookahead = int(lag)

        # precompute DOM "index returns": median of DOM changes (or raw returns before DOM valid)
        for ibar in range(self.N):
            if ibar < L:
                self.dom_index_returns[ibar] = np.median(self.returns[:, ibar])
            else:
                delta_dom = self.dom[ibar] - self.dom[ibar - 1]
                self.dom_index_returns[ibar] = np.median(delta_dom)

        for ibar in range(L - 1, self.N):
            idx = self.dom_index_returns[ibar : ibar - L : -1]  # reverse chrono, length L
            w = idx[lag:]
            med = np.median(w)

            idx_off = 1e-30
            idx_def = -1e-30
            for val in w:
                d = val - med
                if d >= 0:
                    idx_off += d
                else:
                    idx_def += d

            rm_row = np.empty(self.M, float)
            for m in range(self.M):
                off = 0.0
                de  = 0.0
                for i in range(lag, L):
                    if (ibar - i) < L:
                        ret = self.returns[m, ibar - i]
                    else:
                        ret = self.dom[ibar - i, m] - self.dom[ibar - i - 1, m]
                    d = ret - med
                    if idx[i] >= med:
                        off += d
                    else:
                        de  += d
                val = 70.710678 * (off / idx_off - de / idx_def)
                val = min(300.0, max(-300.0, val))
                rm_row[m] = val

            if lag == 0:
                self.rm[ibar] = rm_row
            else:
                self.rm_lagged[ibar] = rm_row

            # fractiles
            if lag == 0:
                order = np.argsort(rm_row)
                self.rm_fractile[ibar, order] = np.arange(self.M, dtype=float) / (self.M - 1.0)

    def compute_rs_ps(self) -> None:
        """Leader/Laggard performance from lagged RS (port of compute_rs_ps)."""
        L = self.lookback
        k = int(self.spread_tail * (self.M + 1)) - 1
        k = max(0, k)
        n = k + 1

        for ibar in range(L - 1, self.N):
            row = self.rs_lagged[ibar]
            order = np.argsort(row)
            # laggard group indices: order[:n], leader group: order[-n:]
            lagg = order[:n]
            lead = order[-n:]

            # average next rs_lookahead bars of returns for those groups
            rsum_lead = 0.0
            rsum_lagg = 0.0
            for i in range(self.rs_lookahead):
                rsum_lagg += self.returns[lagg, ibar - i].sum()
                rsum_lead += self.returns[lead, ibar - i].sum()
            self.rs_leader[ibar] = rsum_lead / (n * self.rs_lookahead)
            self.rs_laggard[ibar] = rsum_lagg / (n * self.rs_lookahead)

            # also the universe average next bar (as in C++)
            self.oos_avg[ibar] = self.returns[:, ibar].mean()

    def compute_rm_ps(self) -> None:
        """Leader/Laggard performance from lagged RM (port of compute_rm_ps)."""
        L = self.lookback
        k = int(self.spread_tail * (self.M + 1)) - 1
        k = max(0, k)
        n = k + 1

        for ibar in range(L - 1, self.N):
            row = self.rm_lagged[ibar]
            order = np.argsort(row)
            lagg = order[:n]
            lead = order[-n:]

            rsum_lead = 0.0
            rsum_lagg = 0.0
            for i in range(self.rm_lookahead):
                rsum_lagg += self.returns[lagg, ibar - i].sum()
                rsum_lead += self.returns[lead, ibar - i].sum()
            self.rm_leader[ibar] = rsum_lead / (n * self.rm_lookahead)
            self.rm_laggard[ibar] = rsum_lagg / (n * self.rm_lookahead)

    def compute_CMA(self) -> None:
        """
        CMA scan (port of compute_CMA):
        - Maintain EMA of DOM index for each period in [min_CMA..max_CMA]
        - Pick period that maximizes in-sample equity using universe avg,
          then apply it one-step OOS to universe and leaders.
        """
        L = self.lookback
        rng = self.max_CMA - self.min_CMA + 1
        self.CMA_alpha[:] = 2.0 / (np.arange(self.min_CMA, self.max_CMA + 1, dtype=float) + 1.0)
        self.CMA_smoothed[:] = 0.0
        self.CMA_equity[:] = 0.0
        self.CMA_OOS[:L+2] = 0.0
        self.CMA_leader_OOS[:L+2] = 0.0

        for ibar in range(L + 2, self.N):
            # pick the period maximizing equity if we bought when dom_index > smoothed
            best_equity = -1e60
            ibest = self.min_CMA
            for i, period in enumerate(range(self.min_CMA, self.max_CMA + 1)):
                if self.dom_index[ibar - 2] > self.CMA_smoothed[i]:
                    self.CMA_equity[i] += self.oos_avg[ibar - 1]
                if self.CMA_equity[i] > best_equity:
                    best_equity = self.CMA_equity[i]
                    ibest = period
                # update smoothed (through ibar-2)
                a = self.CMA_alpha[i]
                self.CMA_smoothed[i] = a * self.dom_index[ibar - 2] + (1.0 - a) * self.CMA_smoothed[i]

            # OOS at ibar using ibest
            i_idx = ibest - self.min_CMA
            if self.dom_index[ibar - 1] > self.CMA_smoothed[i_idx]:
                self.CMA_OOS[ibar] = self.oos_avg[ibar]

                # leader OOS: top tail by RM known at ibar-1
                row = self.rm[ibar - 1]
                order = np.argsort(row)
                k = int(self.spread_tail * (self.M + 1)) - 1
                k = max(0, k)
                n = k + 1
                leaders = order[-n:]
                self.CMA_leader_OOS[ibar] = self.returns[leaders, ibar].mean()

    # --------------- convenience 'getters' (shape match to C++) ---------------

    def get_market_index(self) -> np.ndarray:
        # cumulative sum of index returns, aligned like the C++ getter
        out = np.zeros(self.B, float)
        s = 0.0
        for i in range(self.lookback, self.B):
            s += self.mkt_index_returns[i - 1]
            out[i] = s
        return out

    def get_dom_index(self) -> np.ndarray:
        out = np.zeros(self.B, float); s = 0.0
        for i in range(self.lookback, self.B):
            s += self.dom_index_returns[i - 1]
            out[i] = s
        return out

    def get_array_by_market(self, arrNM: np.ndarray, ord_num: int) -> np.ndarray:
        # ord_num: 1..M, like the C++ (0 means "index" for DOM/DOE getters)
        out = np.zeros(self.B, float)
        for i in range(self.lookback, self.B):
            out[i] = arrNM[i - 1, ord_num - 1]
        return out

    def get_dom(self, ord_num: int) -> np.ndarray:
        out = np.zeros(self.B, float)
        if ord_num == 0:
            for i in range(self.lookback, self.B):
                out[i] = self.dom_index[i - 1]
        else:
            for i in range(self.lookback, self.B):
                out[i] = self.dom[i - 1, ord_num - 1]
        return out

    def get_doe(self, ord_num: int) -> np.ndarray:
        out = np.zeros(self.B, float)
        if ord_num == 0:
            for i in range(self.lookback, self.B):
                out[i] = self.doe_index[i - 1]
        else:
            for i in range(self.lookback, self.B):
                out[i] = self.doe[i - 1, ord_num - 1]
        return out

    # straight copies of cumulative/equity getters
    def _cum_from_row(self, row: np.ndarray) -> np.ndarray:
        out = np.zeros(self.B, float); s = 0.0
        for i in range(self.lookback, self.B):
            s += row[i - 1]
            out[i] = s
        return out

    def get_rss(self) -> np.ndarray:          return self._cum_from_row(self.rss)
    def get_rss_change(self) -> np.ndarray:   return self._cum_from_row(self.rss_change)
    def get_rs(self, ord_num: int) -> np.ndarray:          return self.get_array_by_market(self.rs, ord_num)
    def get_rs_fractile(self, ord_num: int) -> np.ndarray: return self.get_array_by_market(self.rs_fractile, ord_num)
    def get_rm(self, ord_num: int) -> np.ndarray:          return self.get_array_by_market(self.rm, ord_num)
    def get_rm_fractile(self, ord_num: int) -> np.ndarray: return self.get_array_by_market(self.rm_fractile, ord_num)
    def get_rs_leader_equity(self) -> np.ndarray:          return self._cum_from_row(self.rs_leader)
    def get_rs_laggard_equity(self) -> np.ndarray:         return self._cum_from_row(self.rs_laggard)
    def get_rs_ps(self) -> np.ndarray:                      return self._cum_from_row(self.rs_leader - self.rs_laggard)
    def get_rs_leader_advantage(self) -> np.ndarray:        return self._cum_from_row(self.rs_leader - self.oos_avg)
    def get_rs_laggard_advantage(self) -> np.ndarray:       return self._cum_from_row(self.rs_laggard - self.oos_avg)
    def get_rm_leader_equity(self) -> np.ndarray:           return self._cum_from_row(self.rm_leader)
    def get_rm_laggard_equity(self) -> np.ndarray:          return self._cum_from_row(self.rm_laggard)
    def get_rm_ps(self) -> np.ndarray:                      return self._cum_from_row(self.rm_leader - self.rm_laggard)
    def get_oos_avg(self) -> np.ndarray:                    return self._cum_from_row(self.oos_avg)
    def get_CMA_OOS(self) -> np.ndarray:                    return self._cum_from_row(self.CMA_OOS)
    def get_CMA_leader_OOS(self) -> np.ndarray:             return self._cum_from_row(self.CMA_leader_OOS)
