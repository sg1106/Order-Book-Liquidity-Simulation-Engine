"""
orderbook_engine.py
───────────────────
Core limit-order-book matching engine with full microstructure telemetry.

Features
────────
• Multi-level price-time matching for market orders (partial fills)
• Limit / cancel / market order types
• Automatic book healing: spread ≤ 5 ticks at all times
• Tracked deques (rolling windows):
    price_history, spread_history, ofi_history,
    volume_history, returns_history, cum_delta_hist, vwap_history
• Volume profile (price → total traded volume)
• Trade history (Trade dataclass: price, size, side, time)
"""

import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    price: float
    size:  int
    side:  str   # 'buy' = aggressor bought, 'sell' = aggressor sold
    time:  int


# ─────────────────────────────────────────────────────────────────────────────
class OrderBookEngine:
    """Level-2 order book with full telemetry."""

    PRICE_MIN  = 1.0
    PRICE_MAX  = 9_999.0

    def __init__(self, n_levels: int = 30, mid_price: float = 100.0,
                 tick_size: float = 0.1):
        self.n_levels  = n_levels
        self.mid_price = float(mid_price)
        self.tick_size = float(tick_size)

        # Rolling telemetry
        self.trade_history:   deque[Trade] = deque(maxlen=500)
        self.price_history:   deque[float] = deque(maxlen=1_000)
        self.spread_history:  deque[float] = deque(maxlen=1_000)
        self.ofi_history:     deque[float] = deque(maxlen=500)
        self.volume_history:  deque[float] = deque(maxlen=500)
        self.returns_history: deque[float] = deque(maxlen=500)
        self.cum_delta_hist:  deque[float] = deque(maxlen=500)
        self.vwap_history:    deque[float] = deque(maxlen=500)

        # Aggregates
        self.volume_profile: dict[float, int] = {}
        self.total_volume:   int  = 0
        self._cum_delta:     float = 0.0
        self._vwap_num:      float = 0.0
        self._vwap_den:      float = 0.0

        self.reset()

    # ──────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Re-initialise book with exponentially-decaying depth."""
        bp = np.round(
            [self.mid_price - (i + 1) * self.tick_size for i in range(self.n_levels)], 4)
        ap = np.round(
            [self.mid_price + (i + 1) * self.tick_size for i in range(self.n_levels)], 4)

        decay = np.exp(-np.arange(self.n_levels) * 0.12)
        self.bids = pd.Series(
            (np.random.poisson(80, self.n_levels) * decay).clip(1).astype(float),
            index=bp, dtype=float)
        self.asks = pd.Series(
            (np.random.poisson(80, self.n_levels) * decay).clip(1).astype(float),
            index=ap, dtype=float)

        self.time          = 0
        self.total_volume  = 0
        self._cum_delta    = 0.0
        self._vwap_num     = 0.0
        self._vwap_den     = 0.0
        self.volume_profile = {}

        for dq in (self.trade_history, self.price_history, self.spread_history,
                   self.ofi_history, self.volume_history, self.returns_history,
                   self.cum_delta_hist, self.vwap_history):
            dq.clear()

        self._record_price()

    # ──────────────────────────────────────────────────────────────────────
    def _record_price(self) -> None:
        if not len(self.bids) or not len(self.asks):
            return
        bb  = float(self.bids.index.max())
        ba  = float(self.asks.index.min())
        if ba <= bb:
            return
        mid    = (bb + ba) / 2.0
        spread = ba - bb
        mid    = float(np.clip(mid, self.PRICE_MIN, self.PRICE_MAX))
        self.mid_price = mid
        self.price_history.append(mid)
        self.spread_history.append(spread)

    def _record_telemetry(self) -> None:
        """Called once per step after all orders are processed."""
        # OFI (order-flow imbalance at top of book)
        bb_vol = float(self.bids.iloc[0]) if len(self.bids) else 0.0
        ba_vol = float(self.asks.iloc[0]) if len(self.asks) else 0.0
        denom  = bb_vol + ba_vol
        self.ofi_history.append((bb_vol - ba_vol) / denom if denom else 0.0)

        # Cumulative delta snapshot
        self.cum_delta_hist.append(self._cum_delta)

        # VWAP snapshot
        vwap = self._vwap_num / self._vwap_den if self._vwap_den > 0 else self.mid_price
        self.vwap_history.append(float(np.clip(vwap, self.PRICE_MIN, self.PRICE_MAX)))

        # Log return
        ph = list(self.price_history)
        if len(ph) >= 2 and ph[-2] > 0.0 and ph[-1] > 0.0:
            self.returns_history.append(float(np.log(ph[-1] / ph[-2])))

    def _cleanup(self) -> None:
        """Remove depleted levels; heal spreads wider than 5 ticks."""
        self.bids = self.bids[self.bids > 0].sort_index(ascending=False)
        self.asks = self.asks[self.asks > 0].sort_index(ascending=True)

        # Heal wide spreads
        bb = float(self.bids.index.max()) if len(self.bids) else self.mid_price - self.tick_size
        ba = float(self.asks.index.min()) if len(self.asks) else self.mid_price + self.tick_size
        if ba - bb > 5 * self.tick_size:
            nb = round(bb + self.tick_size, 4)
            na = round(ba - self.tick_size, 4)
            if nb < na:
                self.bids[nb] = max(10, int(np.random.poisson(40)))
                self.asks[na] = max(10, int(np.random.poisson(40)))
            else:
                self.bids[round(self.mid_price - self.tick_size, 4)] = max(10, int(np.random.poisson(40)))
                self.asks[round(self.mid_price + self.tick_size, 4)] = max(10, int(np.random.poisson(40)))

        self.bids = self.bids[self.bids > 0].sort_index(ascending=False)
        self.asks = self.asks[self.asks > 0].sort_index(ascending=True)

        # Replenish deep levels
        while len(self.bids) < self.n_levels:
            lo = float(self.bids.index.min()) if len(self.bids) else self.mid_price
            np_ = round(lo - self.tick_size, 4)
            self.bids[np_] = max(1, int(np.random.poisson(25)))
            self.bids = self.bids.sort_index(ascending=False)

        while len(self.asks) < self.n_levels:
            hi = float(self.asks.index.max()) if len(self.asks) else self.mid_price
            np_ = round(hi + self.tick_size, 4)
            self.asks[np_] = max(1, int(np.random.poisson(25)))
            self.asks = self.asks.sort_index(ascending=True)

    # ──────────────────────────────────────────────────────────────────────
    def get_state(self) -> tuple[pd.Series, pd.Series, int]:
        return self.bids.copy(), self.asks.copy(), self.time

    def get_best_bid(self) -> float | None:
        return float(self.bids.index.max()) if len(self.bids) else None

    def get_best_ask(self) -> float | None:
        return float(self.asks.index.min()) if len(self.asks) else None

    # ──────────────────────────────────────────────────────────────────────
    def step(self, order_flow: dict) -> list[Trade]:
        """Process one order; return list of Trade objects (may be empty)."""
        trades: list[Trade] = []

        otype = order_flow['type']
        side  = order_flow.get('side', 'buy')

        if otype == 'market':
            remaining = int(order_flow['size'])
            if side == 'buy':
                book = sorted(self.asks.index)
            else:
                book = sorted(self.bids.index, reverse=True)

            for price in book:
                if remaining <= 0:
                    break
                if side == 'buy':
                    avail  = int(self.asks.get(price, 0))
                    filled = min(avail, remaining)
                    if filled > 0:
                        self.asks[price] -= filled
                else:
                    avail  = int(self.bids.get(price, 0))
                    filled = min(avail, remaining)
                    if filled > 0:
                        self.bids[price] -= filled

                if filled > 0:
                    remaining -= filled
                    t = Trade(float(price), filled, side, self.time)
                    trades.append(t)
                    self.total_volume += filled
                    # Telemetry
                    pk = round(float(price), 1)
                    self.volume_profile[pk] = self.volume_profile.get(pk, 0) + filled
                    self.volume_history.append(float(filled))
                    self._cum_delta  += filled if side == 'buy' else -filled
                    self._vwap_num   += float(price) * filled
                    self._vwap_den   += filled

        elif otype == 'limit':
            price = float(round(order_flow['price'], 1))
            size  = int(order_flow.get('size', 0))
            if size <= 0:
                return trades
            price = float(np.clip(price, self.PRICE_MIN, self.PRICE_MAX))
            if side == 'buy':
                self.bids[price] = self.bids.get(price, 0) + size
                self.bids = self.bids.sort_index(ascending=False)
            else:
                self.asks[price] = self.asks.get(price, 0) + size
                self.asks = self.asks.sort_index(ascending=True)

        elif otype == 'cancel':
            price = float(round(order_flow.get('price', 0), 1))
            size  = int(order_flow.get('size', 0))
            if side == 'buy' and price in self.bids.index:
                self.bids[price] = max(0.0, float(self.bids[price]) - size)
            elif side == 'sell' and price in self.asks.index:
                self.asks[price] = max(0.0, float(self.asks[price]) - size)

        for tr in trades:
            self.trade_history.append(tr)

        self._cleanup()
        self._record_price()
        self._record_telemetry()
        self.time += 1
        return trades
