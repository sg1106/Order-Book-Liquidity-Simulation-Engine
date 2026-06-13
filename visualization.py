"""
visualization.py
────────────────
12 Plotly charts for the Order Book Liquidity Engine.

Chart inventory
───────────────
 1. plot_orderbook_3d          — dual bid/ask liquidity surface over time
 2. plot_depth_chart           — cumulative L2 depth (classic ladder)
 3. plot_price_history         — mid price + Bollinger bands + spread bars
 4. plot_volume_heatmap        — bid-ask imbalance heatmap (price × time)
 5. plot_trade_tape            — scatter of individual executions
 6. plot_ofi_history           — order-flow imbalance bar chart
 7. plot_volume_profile        — horizontal market-profile bars + VWAP
 8. plot_mm_dashboard          — MM PnL and inventory over time
 9. plot_slippage_curve        — price impact vs order size (log scale)
10. plot_regime_price          — price history with regime background shading
11. plot_returns_distribution  — histogram + normal overlay + kurtosis
12. plot_cum_delta             — cumulative buy-minus-sell volume
"""

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ── Design tokens ──────────────────────────────────────────────────────────────
BG        = '#0a0e13'
PANEL     = '#111820'
BORDER    = '#1e2a35'
TXT       = '#c8d8e8'
TXT_DIM   = '#4a6a80'
BID_C     = '#00d4a0'    # teal-green
ASK_C     = '#ff5555'    # coral-red
ACCENT    = '#4d9fff'    # electric blue
WARN      = '#ffb347'    # amber
GRID_C    = '#151f2a'
ZERO_LINE = '#243040'
FONT_MONO = 'JetBrains Mono, Consolas, monospace'

_LAYOUT = dict(
    paper_bgcolor=PANEL,
    plot_bgcolor=BG,
    font=dict(color=TXT, family=FONT_MONO, size=11),
    margin=dict(l=58, r=22, b=48, t=52),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=BORDER,
                borderwidth=1, font=dict(color=TXT, size=10)),
)


def _ax(title: str = '', gridcolor: str = GRID_C, zeroline: bool = True) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=TXT_DIM, size=10)),
        color=TXT, gridcolor=gridcolor, gridwidth=1,
        linecolor=BORDER, showgrid=True,
        zerolinecolor=ZERO_LINE, zerolinewidth=1 if zeroline else 0,
        tickfont=dict(family=FONT_MONO, size=10),
    )


def _empty_fig(message: str = 'Collecting data\u2026', height: int = 300) -> go.Figure:
    """Dark-themed placeholder shown while a chart doesn't yet have enough
    data to render. A bare go.Figure() has no layout at all, so Plotly
    falls back to a white background — this avoids that 'white box'."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5, y=0.5, xref='paper', yref='paper',
        showarrow=False,
        font=dict(color=TXT_DIM, size=12, family=FONT_MONO),
    )
    fig.update_layout(
        **_LAYOUT,
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
        height=height,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1. 3D Liquidity Surface
# ─────────────────────────────────────────────────────────────────────────────
def plot_orderbook_3d(orderbook_history: list, max_snapshots: int = 80) -> go.Figure:
    """Dual surface: bid (teal) and ask (coral) liquidity over time."""
    if not orderbook_history:
        return _empty_fig('Waiting for order book snapshots\u2026', height=520)

    history = orderbook_history[-max_snapshots:]
    times   = [h['time'] for h in history]

    all_prices: set = set()
    for h in history:
        all_prices.update(h['bids'].index.tolist())
        all_prices.update(h['asks'].index.tolist())
    price_levels = sorted(all_prices)

    T, P = len(times), len(price_levels)
    p_idx   = {p: j for j, p in enumerate(price_levels)}
    bid_mat = np.zeros((T, P))
    ask_mat = np.zeros((T, P))
    for i, h in enumerate(history):
        for p, v in h['bids'].items():
            if p in p_idx:
                bid_mat[i, p_idx[p]] = float(v)
        for p, v in h['asks'].items():
            if p in p_idx:
                ask_mat[i, p_idx[p]] = float(v)

    bid_cs = [[0, 'rgba(0,180,140,0.0)'],  [0.4, 'rgba(0,200,150,0.50)'],
              [1, 'rgba(0,240,180,0.95)']]
    ask_cs = [[0, 'rgba(220,60,60,0.0)'],   [0.4, 'rgba(240,70,70,0.50)'],
              [1, 'rgba(255,100,80,0.95)']]

    fig = go.Figure()
    fig.add_trace(go.Surface(
        z=bid_mat, x=price_levels, y=times,
        colorscale=bid_cs, showscale=False, opacity=0.88, name='Bids',
        contours=dict(z=dict(show=True, color='rgba(0,220,160,0.20)', width=1)),
        hovertemplate='Price: %{x:.2f}<br>Time: %{y}<br>Bid Vol: %{z:.0f}<extra>Bids</extra>',
    ))
    fig.add_trace(go.Surface(
        z=ask_mat, x=price_levels, y=times,
        colorscale=ask_cs, showscale=False, opacity=0.88, name='Asks',
        contours=dict(z=dict(show=True, color='rgba(255,80,80,0.20)', width=1)),
        hovertemplate='Price: %{x:.2f}<br>Time: %{y}<br>Ask Vol: %{z:.0f}<extra>Asks</extra>',
    ))

    fig.update_layout(
        **{k: v for k, v in _LAYOUT.items() if k != 'margin'},
        title=dict(text='3D Order Book Liquidity Surface',
                   font=dict(color=TXT, size=14), x=0.02),
        scene=dict(
            xaxis=dict(title='Price',  color=TXT, gridcolor=GRID_C,
                       backgroundcolor=BG, showbackground=True, tickfont=dict(size=9)),
            yaxis=dict(title='Time',   color=TXT, gridcolor=GRID_C,
                       backgroundcolor=BG, showbackground=True, tickfont=dict(size=9)),
            zaxis=dict(title='Volume', color=TXT, gridcolor=GRID_C,
                       backgroundcolor=BG, showbackground=True, tickfont=dict(size=9)),
            bgcolor=BG,
            camera=dict(eye=dict(x=1.55, y=-1.45, z=1.0)),
        ),
        margin=dict(l=0, r=0, b=0, t=42),
        height=520,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cumulative Depth Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_depth_chart(bids: pd.Series, asks: pd.Series) -> go.Figure:
    if len(bids) == 0 or len(asks) == 0:
        return _empty_fig('No order book data\u2026', height=380)

    bp  = sorted(bids.index, reverse=True)
    ap  = sorted(asks.index)
    bc  = np.cumsum([float(bids[p]) for p in bp])
    ac  = np.cumsum([float(asks[p]) for p in ap])
    mid = (max(bp) + min(ap)) / 2.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bp + [min(bp)], y=list(bc) + [0],
        fill='tozeroy', fillcolor='rgba(0,212,160,0.10)',
        line=dict(color=BID_C, width=2), name='Bids', mode='lines',
        hovertemplate='Bid %{x:.3f} — cum vol %{y:,.0f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=ap + [max(ap)], y=list(ac) + [0],
        fill='tozeroy', fillcolor='rgba(255,85,85,0.10)',
        line=dict(color=ASK_C, width=2), name='Asks', mode='lines',
        hovertemplate='Ask %{x:.3f} — cum vol %{y:,.0f}<extra></extra>',
    ))
    fig.add_vline(x=mid, line=dict(color=ACCENT, width=1, dash='dot'))
    fig.add_annotation(
        x=mid, y=0, yref='paper', yanchor='bottom',
        text=f"<b>{mid:.3f}</b>",
        font=dict(color=ACCENT, size=10, family=FONT_MONO),
        showarrow=False, bgcolor='rgba(10,14,19,0.75)',
        bordercolor=ACCENT, borderwidth=1, borderpad=4,
    )
    fig.update_layout(
        **_LAYOUT,
        title=dict(text='Market Depth (Cumulative L2)', font=dict(color=TXT, size=14), x=0.02),
        xaxis=_ax('Price'), yaxis=_ax('Cumulative Volume'),
        height=380,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Price History + Bollinger + Spread
# ─────────────────────────────────────────────────────────────────────────────
def plot_price_history(price_history, spread_history) -> go.Figure:
    prices  = list(price_history)
    spreads = list(spread_history)
    if not prices:
        return _empty_fig('Waiting for price history\u2026', height=400)

    tx  = list(range(len(prices)))
    tsx = list(range(len(spreads)))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.68, 0.32],
        subplot_titles=['Mid Price', 'Bid-Ask Spread'],
        vertical_spacing=0.09,
    )

    # Price line + faint fill
    fig.add_trace(go.Scatter(
        x=tx, y=prices, line=dict(color=ACCENT, width=1.5),
        fill='tozeroy', fillcolor='rgba(77,159,255,0.04)',
        name='Mid Price',
        hovertemplate='t=%{x}  price=%{y:.4f}<extra></extra>',
    ), row=1, col=1)

    # Bollinger bands (±1σ rolling 20)
    if len(prices) > 20:
        s  = pd.Series(prices)
        mu = s.rolling(20).mean()
        sd = s.rolling(20).std()
        fig.add_trace(go.Scatter(
            x=tx + tx[::-1],
            y=(mu + sd).tolist() + (mu - sd).tolist()[::-1],
            fill='toself', fillcolor='rgba(77,159,255,0.05)',
            line=dict(width=0), showlegend=False, hoverinfo='skip',
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=tx, y=mu.tolist(),
            line=dict(color=TXT_DIM, width=1, dash='dot'),
            name='MA-20', hoverinfo='skip',
        ), row=1, col=1)

    # Spread bars coloured by percentile
    if spreads:
        p75 = float(np.percentile(spreads, 75))
        fig.add_trace(go.Bar(
            x=tsx, y=spreads,
            marker=dict(
                color=[WARN if s > p75 else ACCENT for s in spreads],
                opacity=0.65),
            name='Spread',
            hovertemplate='t=%{x}  spread=%{y:.4f}<extra></extra>',
        ), row=2, col=1)

    fig.update_layout(
        **_LAYOUT, height=400,
        title=dict(text='Price & Spread History', font=dict(color=TXT, size=14), x=0.02),
    )
    for r in [1, 2]:
        fig.update_xaxes(_ax('Time'), row=r, col=1)
        fig.update_yaxes(_ax(), row=r, col=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Volume Imbalance Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_volume_heatmap(orderbook_history: list, max_snapshots: int = 80) -> go.Figure:
    if len(orderbook_history) < 2:
        return _empty_fig('Step the simulation to build heatmap history\u2026', height=400)

    history = orderbook_history[-max_snapshots:]
    times   = [h['time'] for h in history]

    all_prices: set = set()
    for h in history:
        all_prices.update(h['bids'].index[:15].tolist())
        all_prices.update(h['asks'].index[:15].tolist())
    price_levels = sorted(all_prices)

    mat = np.zeros((len(price_levels), len(times)))
    for i, h in enumerate(history):
        for j, p in enumerate(price_levels):
            bv = float(h['bids'].get(p, 0))
            av = float(h['asks'].get(p, 0))
            mat[j, i] = bv - av

    cs = [[0.00, '#cc2222'], [0.20, '#661111'],
          [0.45, BG],         [0.55, BG],
          [0.80, '#006644'], [1.00, '#00d4a0']]
    lim = max(1.0, float(np.percentile(np.abs(mat[mat != 0]), 90)) if mat.any() else 1.0)

    fig = go.Figure(go.Heatmap(
        z=mat, x=times,
        y=[f"{p:.2f}" for p in price_levels],
        colorscale=cs, zmin=-lim, zmax=lim, zmid=0,
        colorbar=dict(
            title=dict(text='Bid−Ask', font=dict(color=TXT_DIM, size=10)),
            tickfont=dict(color=TXT_DIM, size=9, family=FONT_MONO),
            thickness=10, len=0.8, bgcolor='rgba(0,0,0,0)',
        ),
        hovertemplate='t:%{x}  price:%{y}  imb:%{z:.1f}<extra></extra>',
        xgap=0.3, ygap=0.3,
    ))
    fig.update_layout(
        **_LAYOUT,
        title=dict(text='Volume Imbalance Heatmap (Bid − Ask)', font=dict(color=TXT, size=14), x=0.02),
        xaxis=_ax('Time Step'),
        yaxis=dict(title=dict(text='Price', font=dict(color=TXT_DIM, size=10)),
                   color=TXT, tickfont=dict(size=9, family=FONT_MONO), autorange=True),
        height=400,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Trade Tape
# ─────────────────────────────────────────────────────────────────────────────
def plot_trade_tape(trade_history) -> go.Figure:
    trades = list(trade_history)[-200:]
    if not trades:
        return _empty_fig('No trades yet\u2026', height=350)

    buys  = [t for t in trades if t.side == 'buy']
    sells = [t for t in trades if t.side == 'sell']

    def _scatter(lst, color, name, symbol):
        if not lst:
            return go.Scatter(x=[], y=[], mode='markers', name=name)
        return go.Scatter(
            x=[t.time  for t in lst],
            y=[t.price for t in lst],
            mode='markers',
            marker=dict(color=color,
                        size=[max(5, min(30, t.size // 8)) for t in lst],
                        symbol=symbol, opacity=0.78,
                        line=dict(color='rgba(255,255,255,0.12)', width=0.5)),
            name=name,
            text=[f"{'BUY' if t.side=='buy' else 'SELL'}  {t.size:,} @ {t.price:.3f}" for t in lst],
            hoverinfo='text',
        )

    fig = go.Figure([
        _scatter(buys,  BID_C, '▲ Buy',  'triangle-up'),
        _scatter(sells, ASK_C, '▼ Sell', 'triangle-down'),
    ])
    fig.update_layout(
        **_LAYOUT,
        title=dict(text='Trade Tape', font=dict(color=TXT, size=14), x=0.02),
        xaxis=_ax('Time'), yaxis=_ax('Price'),
        height=350,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Order Flow Imbalance
# ─────────────────────────────────────────────────────────────────────────────
def plot_ofi_history(ofi_history) -> go.Figure:
    vals = list(ofi_history)
    if not vals:
        return _empty_fig('Waiting for order flow data\u2026', height=220)

    fig = go.Figure(go.Bar(
        x=list(range(len(vals))), y=vals,
        marker=dict(color=[BID_C if v > 0 else ASK_C for v in vals], opacity=0.70),
        hovertemplate='t=%{x}  OFI=%{y:.3f}<extra></extra>',
    ))
    fig.add_hline(y=0, line=dict(color=ZERO_LINE, width=1))
    fig.update_layout(
        **_LAYOUT,
        title=dict(text='Order Flow Imbalance (Top of Book)', font=dict(color=TXT, size=14), x=0.02),
        xaxis=_ax('Time'), yaxis=_ax('OFI'),
        height=220, showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Volume Profile (Market Profile)
# ─────────────────────────────────────────────────────────────────────────────
def plot_volume_profile(volume_profile: dict, mid: float) -> go.Figure:
    if not volume_profile:
        return _empty_fig('No trades yet to build a profile\u2026', height=460)

    prices = sorted(volume_profile.keys())
    vols   = [volume_profile[p] for p in prices]
    colors = [BID_C if p <= mid else ASK_C for p in prices]

    fig = go.Figure(go.Bar(
        y=[f"{p:.1f}" for p in prices], x=vols,
        orientation='h',
        marker=dict(color=colors, opacity=0.72,
                    line=dict(color='rgba(255,255,255,0.06)', width=0.3)),
        hovertemplate='Price %{y} — vol %{x:,.0f}<extra></extra>',
    ))

    if vols:
        vwap = sum(p * v for p, v in zip(prices, vols)) / sum(vols)
        fig.add_vline(x=vwap,
                      line=dict(color=ACCENT, width=1.5, dash='dot'),
                      annotation_text=f"  VWAP {vwap:.2f}",
                      annotation_font=dict(color=ACCENT, size=10),
                      annotation_position="top")

    fig.update_layout(
        **_LAYOUT,
        title=dict(text='Volume Profile (Market Profile)', font=dict(color=TXT, size=14), x=0.02),
        xaxis=_ax('Traded Volume'),
        yaxis=dict(title=dict(text='Price', font=dict(color=TXT_DIM, size=10)),
                   color=TXT, tickfont=dict(size=9, family=FONT_MONO),
                   autorange=True, showgrid=False),
        height=460, bargap=0.10,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. Market Maker Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def plot_mm_dashboard(pnl_history, inv_history) -> go.Figure:
    pnl = list(pnl_history)
    inv = list(inv_history)
    if not pnl:
        return _empty_fig('Step the simulation to build MM history\u2026', height=420)

    tx = list(range(len(pnl)))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.45],
        subplot_titles=['Mark-to-Market PnL', 'Inventory'],
        vertical_spacing=0.10,
    )

    fig.add_trace(go.Scatter(
        x=tx, y=pnl, line=dict(color=ACCENT, width=1.5),
        fill='tozeroy', fillcolor='rgba(77,159,255,0.06)',
        name='PnL',
        hovertemplate='t=%{x}  PnL=%{y:,.2f}<extra></extra>',
    ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=ZERO_LINE, width=1), row=1, col=1)

    fig.add_trace(go.Bar(
        x=tx, y=inv,
        marker=dict(
            color=[BID_C if v > 0 else (ASK_C if v < 0 else TXT_DIM) for v in inv],
            opacity=0.65),
        name='Inventory',
        hovertemplate='t=%{x}  inv=%{y:,}<extra></extra>',
    ), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=ZERO_LINE, width=1), row=2, col=1)

    fig.update_layout(
        **_LAYOUT, height=420,
        title=dict(text='Market Maker Dashboard', font=dict(color=TXT, size=14), x=0.02),
        showlegend=False,
    )
    for r in [1, 2]:
        fig.update_xaxes(_ax('Time'), row=r, col=1)
        fig.update_yaxes(_ax(), row=r, col=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. Slippage Curve (Market Impact)
# ─────────────────────────────────────────────────────────────────────────────
def plot_slippage_curve(orderbook) -> go.Figure:
    bids, asks, _ = orderbook.get_state()
    bb = float(bids.index.max()) if len(bids) else None
    ba = float(asks.index.min()) if len(asks) else None
    if bb is None or ba is None:
        return _empty_fig('No order book data\u2026', height=360)
    mid = (bb + ba) / 2.0

    sizes = sorted(set(int(s) for s in np.round(np.logspace(1, 4.2, 70)).tolist()))

    buy_slip, sell_slip = [], []
    for target in sizes:
        # Buy slippage
        rem, cost = target, 0.0
        for p in sorted(asks.index):
            v = float(asks[p]); f = min(v, rem); cost += p * f; rem -= f
            if rem <= 0: break
        filled = target - rem
        buy_slip.append((cost / filled if filled else ba) - mid)

        # Sell slippage
        rem, proc = target, 0.0
        for p in sorted(bids.index, reverse=True):
            v = float(bids[p]); f = min(v, rem); proc += p * f; rem -= f
            if rem <= 0: break
        filled = target - rem
        sell_slip.append(mid - (proc / filled if filled else bb))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sizes, y=buy_slip, mode='lines',
        line=dict(color=ASK_C, width=2), name='Buy slippage',
        fill='tozeroy', fillcolor='rgba(255,85,85,0.07)',
        hovertemplate='Size %{x:,} → slip %{y:.4f}<extra>Buy</extra>',
    ))
    fig.add_trace(go.Scatter(
        x=sizes, y=sell_slip, mode='lines',
        line=dict(color=BID_C, width=2), name='Sell slippage',
        fill='tozeroy', fillcolor='rgba(0,212,160,0.07)',
        hovertemplate='Size %{x:,} → slip %{y:.4f}<extra>Sell</extra>',
    ))
    fig.update_layout(
        **_LAYOUT,
        title=dict(text='Slippage Curve (Market Impact Function)', font=dict(color=TXT, size=14), x=0.02),
        xaxis=dict(**_ax('Order Size'), type='log'),
        yaxis=_ax('Avg Price Deviation from Mid'),
        height=360,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 10. Price History + Regime Shading
# ─────────────────────────────────────────────────────────────────────────────
def plot_regime_price(orderbook_history: list, price_history) -> go.Figure:
    prices  = list(price_history)
    regimes = [h.get('regime', 'normal') for h in orderbook_history]
    if not prices:
        return _empty_fig('Waiting for price history\u2026', height=360)

    tx = list(range(len(prices)))
    RCOL = {
        'low':    'rgba(77,159,255,0.09)',
        'normal': 'rgba(0,212,160,0.05)',
        'high':   'rgba(255,85,85,0.13)',
    }

    fig = go.Figure()

    # Regime background bands
    n = min(len(tx), len(regimes))
    if n > 0:
        start, cur = 0, regimes[0]
        for i in range(1, n):
            if regimes[i] != cur or i == n - 1:
                fig.add_vrect(
                    x0=start, x1=i,
                    fillcolor=RCOL.get(cur, 'rgba(0,0,0,0)'),
                    layer='below', line_width=0,
                )
                start, cur = i, regimes[i]

    # Price line
    fig.add_trace(go.Scatter(
        x=tx, y=prices, line=dict(color=ACCENT, width=1.5),
        name='Mid Price',
        hovertemplate='t=%{x}  %{y:.4f}<extra></extra>',
    ))

    # MA-20
    if len(prices) > 20:
        ma = pd.Series(prices).rolling(20).mean().tolist()
        fig.add_trace(go.Scatter(
            x=tx, y=ma,
            line=dict(color=WARN, width=1, dash='dot'),
            name='MA-20', hoverinfo='skip',
        ))

    # Regime legend annotations
    for label, color in [('🔵 Low', '#4d9fff'), ('🟢 Normal', '#00d4a0'), ('🔴 High Vol', '#ff5555')]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(color=color, size=10, symbol='square'),
            name=label, showlegend=True,
        ))

    fig.update_layout(
        **_LAYOUT,
        title=dict(text='Price History + Volatility Regime Annotations',
                   font=dict(color=TXT, size=14), x=0.02),
        xaxis=_ax('Time'), yaxis=_ax('Mid Price'),
        height=360,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 11. Returns Distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_returns_distribution(returns_history) -> go.Figure:
    from scipy import stats as sp_stats

    rets = list(returns_history)
    if len(rets) < 15:
        return _empty_fig(f'Need \u2265 15 return samples (have {len(rets)})\u2026', height=340)

    r     = np.array(rets, dtype=float)
    mu, s = r.mean(), r.std()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r, nbinsx=55,
        marker=dict(color=ACCENT, opacity=0.55,
                    line=dict(color='rgba(255,255,255,0.07)', width=0.3)),
        name='Log returns', histnorm='probability density',
        hovertemplate='ret=%{x:.5f}  dens=%{y:.3f}<extra></extra>',
    ))

    xs = np.linspace(r.min(), r.max(), 200)
    fig.add_trace(go.Scatter(
        x=xs, y=sp_stats.norm.pdf(xs, mu, s),
        line=dict(color=WARN, width=2, dash='dash'),
        name='Normal fit', hoverinfo='skip',
    ))

    ek = float(sp_stats.kurtosis(r, fisher=True))
    fig.add_annotation(
        x=0.97, y=0.95, xref='paper', yref='paper',
        text=(f"μ = {mu*1e4:.3f} bps<br>"
              f"σ = {s*1e4:.3f} bps<br>"
              f"Excess kurt = {ek:.2f}"),
        font=dict(color=TXT_DIM, size=10, family=FONT_MONO),
        align='right', showarrow=False,
        bgcolor='rgba(10,14,19,0.80)',
        bordercolor=BORDER, borderwidth=1, borderpad=8,
    )

    fig.update_layout(
        **_LAYOUT,
        title=dict(text='Return Distribution & Fat Tails',
                   font=dict(color=TXT, size=14), x=0.02),
        xaxis=_ax('Log Return'), yaxis=_ax('Density'),
        height=340, barmode='overlay',
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 12. Cumulative Delta
# ─────────────────────────────────────────────────────────────────────────────
def plot_cum_delta(cum_delta_hist) -> go.Figure:
    vals = list(cum_delta_hist)
    if not vals:
        return _empty_fig('Waiting for trade flow\u2026', height=240)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(vals))), y=vals,
        line=dict(color=ACCENT, width=1.5),
        fill='tozeroy', fillcolor='rgba(77,159,255,0.07)',
        name='Cum Δ',
        hovertemplate='t=%{x}  Δ=%{y:+,.0f}<extra></extra>',
    ))
    fig.add_hline(y=0, line=dict(color=ZERO_LINE, width=1))

    fig.update_layout(
        **_LAYOUT,
        title=dict(text='Cumulative Order Flow Delta (Buy Vol − Sell Vol)',
                   font=dict(color=TXT, size=14), x=0.02),
        xaxis=_ax('Time'), yaxis=_ax('Net Volume'),
        height=240, showlegend=False,
    )
    return fig