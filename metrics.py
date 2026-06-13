"""
metrics.py
──────────
Computes and displays a full microstructure metrics panel.

Metrics computed
────────────────
Price          : Mid, Best Bid, Best Ask, VWAP
Spread         : Absolute, bps
Flow           : OFI (full book), Top-of-Book Imbalance, Cumulative Delta
Depth          : Liquidity within ±5t and ±10t
Slippage       : Avg-price deviation for 100 / 500 / 1 000 unit market orders
Volatility     : Realised vol (full window), RV-20 (rolling 20 periods)
Microstructure : Kyle's λ (price impact / unit volume)
                 Amihud illiquidity ratio
                 Return autocorrelation at lag-1
                 Sharpe-like ratio
Volume         : Total traded, book bid vol, book ask vol
"""

import numpy as np
import pandas as pd
try:
    import streamlit as st
    _HAS_ST = True
except ImportError:
    _HAS_ST = False


# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(orderbook) -> dict:
    bids, asks, _t = orderbook.get_state()

    best_bid = float(bids.index.max()) if len(bids) else np.nan
    best_ask = float(asks.index.min()) if len(asks) else np.nan
    spread   = best_ask - best_bid \
               if not (np.isnan(best_bid) or np.isnan(best_ask)) else np.nan
    mid      = (best_ask + best_bid) / 2.0 if not np.isnan(spread) else np.nan

    total_bid_vol = float(bids.sum())
    total_ask_vol = float(asks.sum())
    denom_full    = total_bid_vol + total_ask_vol
    ofi           = (total_bid_vol - total_ask_vol) / denom_full \
                    if denom_full else 0.0

    top_bid  = float(bids.iloc[0]) if len(bids) else 0.0
    top_ask  = float(asks.iloc[0]) if len(asks) else 0.0
    top_d    = top_bid + top_ask
    top_imb  = (top_bid - top_ask) / top_d if top_d else 0.0

    tick = orderbook.tick_size

    # ── Depth within N ticks ──────────────────────────────────────────────
    depth5 = depth10 = 0.0
    if not np.isnan(mid):
        depth5  = float(bids[bids.index >= mid - 5  * tick].sum()
                      + asks[asks.index <= mid + 5  * tick].sum())
        depth10 = float(bids[bids.index >= mid - 10 * tick].sum()
                      + asks[asks.index <= mid + 10 * tick].sum())

    # ── Slippage (buy side) ───────────────────────────────────────────────
    def _slip_buy(target: int) -> float:
        rem, cost = target, 0.0
        for p in sorted(asks.index):
            v = float(asks[p]); f = min(v, rem); cost += p * f; rem -= f
            if rem <= 0: break
        filled = target - rem
        avg = cost / filled if filled else (best_ask if not np.isnan(best_ask) else 0.0)
        return avg - mid if not np.isnan(mid) else 0.0

    # ── Realised volatility ───────────────────────────────────────────────
    prices = list(orderbook.price_history)
    rv_full = rv20 = np.nan
    if len(prices) > 10:
        log_rets = np.diff(np.log(np.array(prices, dtype=float)))
        log_rets = log_rets[np.isfinite(log_rets)]
        if len(log_rets):
            rv_full = float(np.std(log_rets) * np.sqrt(252 * 390))
    if len(prices) > 22:
        lr20 = np.diff(np.log(np.array(prices[-22:], dtype=float)))
        lr20 = lr20[np.isfinite(lr20)]
        if len(lr20):
            rv20 = float(np.std(lr20) * np.sqrt(252 * 390))

    # ── Kyle's lambda ─────────────────────────────────────────────────────
    kylamb = np.nan
    trades = list(orderbook.trade_history)
    if len(trades) >= 5 and len(prices) >= 5:
        w      = min(30, len(trades))
        vol_w  = sum(tr.size for tr in trades[-w:])
        p_move = abs(prices[-1] - prices[max(0, len(prices) - w)])
        kylamb = p_move / vol_w if vol_w else 0.0

    # ── Amihud illiquidity ────────────────────────────────────────────────
    amihud = np.nan
    if len(prices) > 5:
        vols_ = list(orderbook.volume_history)[-50:]
        if len(prices) > 51:
            rabs  = np.abs(np.diff(np.log(np.array(prices[-52:], dtype=float))))
        else:
            rabs  = np.abs(np.diff(np.log(np.array(prices, dtype=float))))
        rabs  = rabs[np.isfinite(rabs)]
        n_    = min(len(rabs), len(vols_))
        if n_ > 0:
            amihud = float(np.mean(rabs[:n_]
                                   / (np.array(vols_[:n_], float) + 1e-9)))

    # ── Sharpe-like ratio ─────────────────────────────────────────────────
    sharpe = np.nan
    if len(prices) > 20:
        r_ = np.diff(np.array(prices, dtype=float))
        r_ = r_[np.isfinite(r_)]
        if len(r_):
            sharpe = float(np.mean(r_) / (np.std(r_) + 1e-12) * np.sqrt(len(r_)))

    # ── Return autocorrelation at lag-1 ───────────────────────────────────
    autocorr = np.nan
    rets_raw = [x for x in list(orderbook.returns_history) if np.isfinite(x)]
    if len(rets_raw) > 10:
        r_arr = np.array(rets_raw)
        if r_arr.std() > 1e-10:          # guard: skip when variance is (near) zero
            cc = np.corrcoef(r_arr[:-1], r_arr[1:])
            if np.isfinite(cc[0, 1]):
                autocorr = float(cc[0, 1])

    # ── VWAP and cumulative delta ─────────────────────────────────────────
    vwap_list = list(orderbook.vwap_history)
    vwap      = vwap_list[-1] if vwap_list else np.nan
    cd_list   = list(orderbook.cum_delta_hist)
    cum_delta = cd_list[-1] if cd_list else 0.0

    spread_bps = ((spread / mid) * 10_000
                  if (not np.isnan(spread) and not np.isnan(mid) and mid > 0)
                  else np.nan)

    return {
        'Mid Price':     mid,
        'Best Bid':      best_bid,
        'Best Ask':      best_ask,
        'Spread':        spread,
        'Spread (bps)':  spread_bps,
        'OFI':           ofi,
        'Top Imbalance': top_imb,
        'Total Bid Vol': total_bid_vol,
        'Total Ask Vol': total_ask_vol,
        'Depth (±5t)':   depth5,
        'Depth (±10t)':  depth10,
        'Slip (100)':    _slip_buy(100),
        'Slip (500)':    _slip_buy(500),
        'Slip (1000)':   _slip_buy(1000),
        'Realized Vol':  rv_full,
        'RV-20':         rv20,
        "Kyle's λ":      kylamb,
        'Amihud':        amihud,
        'Sharpe':        sharpe,
        'Autocorr':      autocorr,
        'VWAP':          vwap,
        'Cum Delta':     cum_delta,
        'Total Volume':  orderbook.total_volume,
    }


# ─────────────────────────────────────────────────────────────────────────────
def _fmt(v, dec: int = 4, prefix: str = '', suffix: str = '') -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return '—'
    return f"{prefix}{v:,.{dec}f}{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
def display_metrics_panel(metrics: dict) -> None:
    if not _HAS_ST:
        return
    mid     = metrics.get('Mid Price', np.nan)
    ofi     = metrics.get('OFI', 0.0)
    top_imb = metrics.get('Top Imbalance', 0.0)

    st.markdown("#### 📊 Live Metrics")

    # ── Price ────────────────────────────────────────────────────────────
    st.metric("Mid Price", _fmt(mid, 3, '$'))
    c1, c2 = st.columns(2)
    c1.metric("Best Bid", _fmt(metrics.get('Best Bid'), 3, '$'))
    c2.metric("Best Ask", _fmt(metrics.get('Best Ask'), 3, '$'))
    vwap = metrics.get('VWAP', np.nan)
    if not (isinstance(vwap, float) and (np.isnan(vwap) or np.isinf(vwap))):
        st.metric("VWAP", _fmt(vwap, 3, '$'))
    st.divider()

    # ── Spread ───────────────────────────────────────────────────────────
    sp_bps = metrics.get('Spread (bps)', np.nan)
    delta  = f"{_fmt(sp_bps, 1)} bps" \
             if not (isinstance(sp_bps, float) and np.isnan(sp_bps)) else None
    st.metric("Spread", _fmt(metrics.get('Spread'), 4), delta)
    st.divider()

    # ── Flow & imbalance ─────────────────────────────────────────────────
    ofi_lbl = "↑ Buy pressure" if ofi > 0.1 else ("↓ Sell pressure" if ofi < -0.1 else "Balanced")
    st.metric("Book OFI",        _fmt(ofi, 3),     ofi_lbl)
    st.metric("Top-of-Book Imb", _fmt(top_imb, 3))
    cd     = metrics.get('Cum Delta', 0)
    cd_lbl = f"{'Net buy' if cd > 0 else 'Net sell'} {abs(int(cd)):,}"
    st.metric("Cum Delta", f"{int(cd):+,}", cd_lbl)
    st.divider()

    # ── Depth & slippage ─────────────────────────────────────────────────
    st.metric("Depth ±5t",    _fmt(metrics.get('Depth (±5t)'),   0, suffix=' u'))
    st.metric("Depth ±10t",   _fmt(metrics.get('Depth (±10t)'),  0, suffix=' u'))
    st.metric("Slip ×100",    _fmt(metrics.get('Slip (100)'),    4))
    st.metric("Slip ×500",    _fmt(metrics.get('Slip (500)'),    4))
    st.metric("Slip ×1 000",  _fmt(metrics.get('Slip (1000)'),   4))
    st.divider()

    # ── Microstructure ───────────────────────────────────────────────────
    rv   = metrics.get('Realized Vol', np.nan)
    rv20 = metrics.get('RV-20',        np.nan)
    rv20_delta = f"RV-20: {_fmt(rv20, 4)}" \
                 if not (isinstance(rv20, float) and np.isnan(rv20)) else None
    st.metric("Realized Vol",   _fmt(rv, 4), rv20_delta)
    st.metric("Kyle's λ",       _fmt(metrics.get("Kyle's λ"), 6))
    st.metric("Amihud",         _fmt(metrics.get('Amihud'),    8))
    ac     = metrics.get('Autocorr', np.nan)
    ac_lbl = None
    if isinstance(ac, float) and np.isfinite(ac):
        ac_lbl = "Mean-reverting" if ac < -0.05 else ("Trending" if ac > 0.05 else "Random walk")
    st.metric("Return Autocorr", _fmt(ac, 3), ac_lbl)
    st.metric("Sharpe",          _fmt(metrics.get('Sharpe'), 3))
    st.divider()

    # ── Volume ───────────────────────────────────────────────────────────
    st.metric("Traded Vol",   f"{int(metrics.get('Total Volume', 0)):,}")
    st.metric("Book Bid Vol", f"{int(metrics.get('Total Bid Vol', 0)):,}")
    st.metric("Book Ask Vol", f"{int(metrics.get('Total Ask Vol', 0)):,}")
