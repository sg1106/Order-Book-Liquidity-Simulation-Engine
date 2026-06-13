"""
simulation.py
─────────────
Realistic market microstructure simulator on top of OrderBookEngine.

Components
──────────
• HawkesProcess          — self-exciting order clustering
• MarketMaker            — Avellaneda-Stoikov with inventory skew + full PnL
• VolatilityRegimeModel  — 3-state Markov chain (Low / Normal / High)
• OrderBookSimulation    — ties it all together; supports 6 scenario presets
"""

import numpy as np
from collections import deque
from orderbook_engine import OrderBookEngine


# ─────────────────────────────────────────────────────────────────────────────
# Scenario presets
# ─────────────────────────────────────────────────────────────────────────────
SCENARIOS: dict[str, dict] = {
    "🔵  Quiet Market":  dict(
        volatility=0.005, order_rate=3,  market_order_size=50,
        hawkes_alpha=0.30, hawkes_beta=2.0,
        regime_high_prob=0.01, mean_rev=0.06, trend_drift=0.0),

    "🟢  Normal Market": dict(
        volatility=0.030, order_rate=6,  market_order_size=100,
        hawkes_alpha=0.55, hawkes_beta=1.2,
        regime_high_prob=0.05, mean_rev=0.025, trend_drift=0.0),

    "🟡  Trending Bull": dict(
        volatility=0.020, order_rate=8,  market_order_size=150,
        hawkes_alpha=0.65, hawkes_beta=1.0,
        regime_high_prob=0.04, mean_rev=0.005, trend_drift=+0.0003),

    "🔴  Bear Panic":    dict(
        volatility=0.060, order_rate=14, market_order_size=300,
        hawkes_alpha=0.80, hawkes_beta=0.9,
        regime_high_prob=0.25, mean_rev=0.05, trend_drift=-0.0005),

    "⚡  Flash Crash":   dict(
        volatility=0.100, order_rate=20, market_order_size=600,
        hawkes_alpha=0.90, hawkes_beta=0.7,
        regime_high_prob=0.55, mean_rev=0.10, trend_drift=-0.0010),

    "🌪️  Crisis":        dict(
        volatility=0.080, order_rate=16, market_order_size=400,
        hawkes_alpha=0.85, hawkes_beta=0.8,
        regime_high_prob=0.40, mean_rev=0.05, trend_drift=0.0),
}


# ─────────────────────────────────────────────────────────────────────────────
class HawkesProcess:
    """Univariate Hawkes process — recent trades excite future order rate."""

    def __init__(self, baseline: float = 0.6, alpha: float = 0.55, beta: float = 1.2):
        self.baseline     = baseline
        self.alpha        = alpha
        self.beta         = beta
        self.intensity    = baseline
        self.last_event_t = 0

    def update(self, t: float, had_event: bool = False) -> float:
        dt = max(float(t) - self.last_event_t, 1e-9)
        self.intensity = self.baseline + (self.intensity - self.baseline) * np.exp(-self.beta * dt)
        if had_event:
            self.intensity   += self.alpha
            self.last_event_t = float(t)
        return max(self.baseline, self.intensity)


# ─────────────────────────────────────────────────────────────────────────────
class MarketMaker:
    """
    Avellaneda-Stoikov market maker.

    Quotes straddle the mid with a half-spread that widens as inventory
    grows (inventory skew + reservation spread).  Full cash/inventory
    PnL is tracked via mark-to-market.
    """

    def __init__(self, inventory_limit: int = 1_000, risk_aversion: float = 0.005):
        self.inventory_limit = inventory_limit
        self.risk_aversion   = risk_aversion
        self.inventory       = 0
        self.cash            = 0.0
        self.last_mid        = None
        self.pnl_history:    deque[float] = deque(maxlen=500)
        self.inv_history:    deque[int]   = deque(maxlen=500)

    @property
    def mark_to_market(self) -> float:
        return self.cash + self.inventory * (self.last_mid or 100.0)

    def get_quotes(self, mid: float, tick: float, vol: float):
        """Return (bid_price, ask_price, quote_size)."""
        reservation = self.risk_aversion * abs(self.inventory) * vol * mid
        half        = max(tick, tick * 1.5 + reservation)
        skew        = self.risk_aversion * self.inventory * mid * 0.4
        bid  = float(np.round(mid - half - skew, 1))
        ask  = float(np.round(mid + half - skew, 1))
        size = max(10, 150 - abs(self.inventory) // 6)
        return bid, ask, int(size)

    def record_trade(self, side: str, size: int, price: float) -> None:
        """Update cash + inventory when an aggressor hits an MM quote."""
        if side == 'buy':               # aggressor bought → MM sold
            self.cash      += price * size
            self.inventory -= size
        else:                           # aggressor sold → MM bought
            self.cash      -= price * size
            self.inventory += size
        self.inventory = int(np.clip(self.inventory,
                                     -self.inventory_limit, self.inventory_limit))

    def snapshot(self, mid: float) -> None:
        self.last_mid = mid
        self.pnl_history.append(self.mark_to_market)
        self.inv_history.append(self.inventory)

    def reset(self) -> None:
        self.inventory = 0
        self.cash      = 0.0
        self.last_mid  = None
        self.pnl_history.clear()
        self.inv_history.clear()


# ─────────────────────────────────────────────────────────────────────────────
class VolatilityRegimeModel:
    """Three-state Markov chain for volatility regimes."""

    REGIMES     = ['low', 'normal', 'high']
    MULTIPLIERS = {'low': 0.25, 'normal': 1.0, 'high': 3.5}

    def __init__(self, high_prob: float = 0.05):
        hp = float(np.clip(high_prob, 0.01, 0.80))
        lp = max(0.01, 0.05 - hp * 0.3)
        mp = max(0.01, 1.0 - hp - lp)
        T  = np.array([
            [0.90,        lp * 2,  hp * 0.5],
            [lp,          0.90,    hp       ],
            [0.02,        mp * 2,  0.85     ],
        ], dtype=float)
        # Normalise rows to sum to 1
        self.TRANSITION = (T.T / T.sum(axis=1)).T
        self.state      = 'normal'
        self.history:   deque[str] = deque(maxlen=1_000)

    def step(self) -> str:
        idx        = self.REGIMES.index(self.state)
        self.state = np.random.choice(self.REGIMES, p=self.TRANSITION[idx])
        self.history.append(self.state)
        return self.state

    @property
    def multiplier(self) -> float:
        return self.MULTIPLIERS[self.state]


# ─────────────────────────────────────────────────────────────────────────────
class OrderBookSimulation:
    """
    Full market simulator.

    Each call to step() generates one time-step of order flow:
      • Hawkes-modulated number of orders
      • 28 % market orders (with MM-inventory / trend bias)
      • 52 % limit orders  (informed 12 %, uninformed 40 %)
      • 20 % cancellations (exponentially distributed depth index)
      • MM requotes every 2 steps
      • Mean-reversion / trend injected as a small directional flow
    """

    def __init__(self, volatility: float = 0.030, order_rate: int = 6,
                 market_order_size: int = 100, scenario: str | None = None):
        cfg = SCENARIOS.get(scenario, {}) if scenario else {}

        self.volatility        = cfg.get('volatility',        volatility)
        self.order_rate        = cfg.get('order_rate',        order_rate)
        self.market_order_size = cfg.get('market_order_size', market_order_size)
        self._trend_drift      = cfg.get('trend_drift',       0.0)
        self._mean_rev         = cfg.get('mean_rev',          0.025)
        self._hawkes_alpha     = cfg.get('hawkes_alpha',      0.55)
        self._hawkes_beta      = cfg.get('hawkes_beta',       1.2)
        self._regime_high_prob = cfg.get('regime_high_prob',  0.05)

        self.orderbook         = OrderBookEngine()
        self.orderbook_history: list[dict] = []
        self.hawkes            = HawkesProcess(alpha=self._hawkes_alpha,
                                               beta=self._hawkes_beta)
        self.market_maker      = MarketMaker()
        self.regime_model      = VolatilityRegimeModel(high_prob=self._regime_high_prob)

        self.t = 0
        self.reset()

    # ──────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        self.orderbook.reset()
        self.orderbook_history = []
        self.hawkes        = HawkesProcess(alpha=self._hawkes_alpha,
                                           beta=self._hawkes_beta)
        self.market_maker.reset()
        self.regime_model  = VolatilityRegimeModel(high_prob=self._regime_high_prob)
        self.t             = 0
        self._record_state()

    def update_params(self, volatility: float, order_rate: int,
                      market_order_size: int) -> None:
        self.volatility        = volatility
        self.order_rate        = order_rate
        self.market_order_size = market_order_size

    # ──────────────────────────────────────────────────────────────────────
    def step(self) -> None:
        self.regime_model.step()
        eff_vol = self.volatility * self.regime_model.multiplier
        mid     = self.orderbook.mid_price
        tick    = self.orderbook.tick_size

        # Hawkes-modulated order count
        intensity = self.hawkes.update(self.t)
        n_orders  = int(np.clip(self.order_rate * intensity, 1, self.order_rate * 4))

        self.market_maker.snapshot(mid)

        for _ in range(n_orders):
            r = np.random.random()

            # ── Market order (28 %) ──────────────────────────────────────
            if r < 0.28:
                inv_bias   = float(np.clip(self.market_maker.inventory / 600, -0.35, 0.35))
                trend_bias = float(np.clip(self._trend_drift * 500, -0.25, 0.25))
                p_buy      = 0.50 + inv_bias * 0.4 + trend_bias
                side       = 'buy' if np.random.random() < p_buy else 'sell'
                size       = max(1, int(np.random.lognormal(
                    np.log(max(1, self.market_order_size)), 0.55)))
                trades     = self.orderbook.step({'type': 'market', 'side': side, 'size': size})
                for tr in trades:
                    self.market_maker.record_trade(tr.side, tr.size, tr.price)
                if trades:
                    self.hawkes.update(self.t, had_event=True)

            # ── Limit order (52 %) ───────────────────────────────────────
            elif r < 0.80:
                side     = np.random.choice(['buy', 'sell'])
                informed = np.random.random() < 0.12
                bb = self.orderbook.get_best_bid() or (mid - tick)
                ba = self.orderbook.get_best_ask() or (mid + tick)

                if side == 'buy':
                    base  = (ba + tick * np.random.normal(-2, eff_vol * 15)
                             if informed else bb)
                    off   = abs(int(np.random.exponential(3))) * tick
                    price = float(np.clip(round(base - off, 1),
                                          mid - 15 * tick, ba - tick))
                else:
                    base  = (bb - tick * np.random.normal(-2, eff_vol * 15)
                             if informed else ba)
                    off   = abs(int(np.random.exponential(3))) * tick
                    price = float(np.clip(round(base + off, 1),
                                          bb + tick, mid + 15 * tick))

                size = max(1, int(np.random.lognormal(3.0, 0.75)))
                self.orderbook.step({'type': 'limit', 'side': side,
                                     'price': price, 'size': size})

            # ── Cancel (20 %) ────────────────────────────────────────────
            else:
                side = np.random.choice(['buy', 'sell'])
                book = self.orderbook.bids if side == 'buy' else self.orderbook.asks
                if len(book) > 3:
                    idx_c = min(len(book) - 1, int(np.random.exponential(3)))
                    price = float(book.index[idx_c])
                    size  = max(1, int(np.random.lognormal(2.8, 0.6)))
                    self.orderbook.step({'type': 'cancel', 'side': side,
                                         'price': price, 'size': size})

        # ── MM requotes every 2 steps ────────────────────────────────────
        if self.t % 2 == 0:
            cur_mid  = self.orderbook.mid_price
            bid, ask, sz = self.market_maker.get_quotes(cur_mid, tick, eff_vol)
            bb = self.orderbook.get_best_bid() or (cur_mid - tick)
            ba = self.orderbook.get_best_ask() or (cur_mid + tick)
            bid = min(bid, bb)
            ask = max(ask, ba)
            if bid < ask:
                self.orderbook.step({'type': 'limit', 'side': 'buy',
                                     'price': bid, 'size': sz})
                self.orderbook.step({'type': 'limit', 'side': 'sell',
                                     'price': ask, 'size': sz})

        # ── Mean-reversion + trend as small directional market orders ────
        deviation = self.orderbook.mid_price - 100.0
        combined  = -deviation * self._mean_rev + self._trend_drift
        if combined > 0.004:
            self.orderbook.step({'type': 'market', 'side': 'buy',
                                 'size': max(1, int(abs(combined) * 500))})
        elif combined < -0.004:
            self.orderbook.step({'type': 'market', 'side': 'sell',
                                 'size': max(1, int(abs(combined) * 500))})

        self.t += 1
        self._record_state()

    # ──────────────────────────────────────────────────────────────────────
    def _record_state(self) -> None:
        bids, asks, t = self.orderbook.get_state()
        self.orderbook_history.append({
            'bids':   bids.copy(),
            'asks':   asks.copy(),
            'time':   t,
            'regime': self.regime_model.state,
            'mm_pnl': self.market_maker.mark_to_market,
        })
        # Rolling window keeps memory bounded
        if len(self.orderbook_history) > 400:
            self.orderbook_history = self.orderbook_history[-400:]
