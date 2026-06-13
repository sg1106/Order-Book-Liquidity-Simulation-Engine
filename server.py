"""
server.py
─────────
Flask backend for the 3D Order Book Liquidity Engine.

Run:   python server.py
Open:  http://localhost:5000

API endpoints
─────────────
GET  /                  → serve index.html
GET  /api/state?tab=X   → full state for active tab (metrics + charts + l2 + trades)
POST /api/step          → {n, tab}  → step N times, return state
POST /api/reset         → {tab}     → reset simulation
POST /api/scenario      → {scenario, tab}
POST /api/inject        → {type, side, size, price?, tab}
GET  /api/export        → CSV download of trade history
"""

from __future__ import annotations
import io, json, os
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file

from simulation import OrderBookSimulation, SCENARIOS
from metrics import compute_metrics
from visualization import (
    plot_orderbook_3d, plot_depth_chart, plot_price_history,
    plot_volume_heatmap, plot_trade_tape, plot_ofi_history,
    plot_volume_profile, plot_mm_dashboard, plot_slippage_curve,
    plot_regime_price, plot_returns_distribution, plot_cum_delta,
)

# ── App & global simulation ────────────────────────────────────────────────────
app  = Flask(__name__, static_folder='.', static_url_path='')
_sim = OrderBookSimulation(scenario='🟢  Normal Market')


# ── Utility helpers ────────────────────────────────────────────────────────────
def _safe(v):
    """Make a value JSON-serialisable (handle nan/inf/numpy scalars)."""
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer, np.floating)):
        return float(v)
    return v

def _safe_dict(d: dict) -> dict:
    return {k: _safe(v) for k, v in d.items()}

import base64


def _decode_bdata(d: dict):
    """Decode Plotly's binary-encoded array format
    {"dtype": "f8", "bdata": "<base64>", "shape": "r, c"?} back into a
    plain (possibly nested) Python list."""
    arr = np.frombuffer(base64.b64decode(d['bdata']), dtype=np.dtype(d['dtype']))
    shape = d.get('shape')
    if shape:
        dims = tuple(int(x) for x in str(shape).split(','))
        arr = arr.reshape(dims)
    return arr.tolist()


def _to_native(obj):
    """Recursively convert numpy/pandas types — and Plotly's 'binary'
    array-encoding dicts ({"dtype": ..., "bdata": ..., "shape": ...}) —
    into plain JSON-safe Python types.

    Plotly 6.x's to_plotly_json()/to_json() encode numeric arrays
    (including 2D z-arrays for heatmaps/surfaces) as base64 'bdata' blobs.
    Without decoding these back to plain lists, several charts — returns
    distribution, volume heatmap, 3D surface, etc. — render blank in the
    browser. We decode bdata back to nested lists and convert any
    remaining numpy scalars/NaN/Inf to native JSON-safe values.
    """
    if isinstance(obj, dict):
        if 'bdata' in obj and 'dtype' in obj:
            return _to_native(_decode_bdata(obj))
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Index):
        return _to_native(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    return obj


def _fig(fig) -> dict:
    """Serialise a Plotly Figure to a plain JSON-safe dict.

    Uses to_plotly_json() + _to_native() instead of to_json() to avoid
    Plotly's binary typed-array encoding, and strips the default 'plotly'
    template (white paper/plot backgrounds, light axis colors, white hover
    labels) so our dark theme isn't overridden anywhere.
    """
    d = _to_native(fig.to_plotly_json())
    d.get('layout', {}).pop('template', None)
    return d


# ── Per-tab chart builder ──────────────────────────────────────────────────────
def _charts_for_tab(tab: str) -> dict:
    """Only compute the charts needed for the active tab."""
    bids, asks, _ = _sim.orderbook.get_state()
    mid = _sim.orderbook.mid_price or 100.0
    out: dict = {}

    if tab == '3d':
        out['3d']       = _fig(plot_orderbook_3d(_sim.orderbook_history))

    elif tab == 'depth':
        out['depth']    = _fig(plot_depth_chart(bids, asks))
        out['slippage'] = _fig(plot_slippage_curve(_sim.orderbook))

    elif tab == 'heatmap':
        out['heatmap']  = _fig(plot_volume_heatmap(_sim.orderbook_history))
        out['ofi']      = _fig(plot_ofi_history(_sim.orderbook.ofi_history))
        out['vprofile'] = _fig(plot_volume_profile(
            _sim.orderbook.volume_profile, mid))

    elif tab == 'analytics':
        out['regime_price'] = _fig(plot_regime_price(
            _sim.orderbook_history, _sim.orderbook.price_history))
        out['tape']         = _fig(plot_trade_tape(_sim.orderbook.trade_history))
        out['cum_delta']    = _fig(plot_cum_delta(_sim.orderbook.cum_delta_hist))

    elif tab == 'mm':
        out['mm'] = _fig(plot_mm_dashboard(
            _sim.market_maker.pnl_history, _sim.market_maker.inv_history))

    elif tab == 'stats':
        out['returns']    = _fig(plot_returns_distribution(
            _sim.orderbook.returns_history))
        out['price_hist'] = _fig(plot_price_history(
            _sim.orderbook.price_history, _sim.orderbook.spread_history))
    # 'l2' tab → no charts
    return out


# ── Rolling stats ──────────────────────────────────────────────────────────────
def _rolling_stats() -> list[dict]:
    rets = [x for x in list(_sim.orderbook.returns_history) if np.isfinite(x)]
    rows = []
    for w in [10, 20, 50, 100, 200]:
        sl = rets[-w:]
        if len(sl) < w:
            continue
        arr = np.array(sl)
        ac  = None
        if arr.std() > 1e-10:
            cc = np.corrcoef(arr[:-1], arr[1:])
            if np.isfinite(cc[0, 1]):
                ac = round(float(cc[0, 1]), 4)
        rows.append({
            'window':      w,
            'mean_bps':    round(float(np.mean(arr)) * 1e4, 3),
            'std_bps':     round(float(np.std(arr))  * 1e4, 3),
            'ann_vol_pct': round(float(np.std(arr))  * np.sqrt(252 * 390) * 100, 2),
            'autocorr':    ac,
            'min_bps':     round(float(np.min(arr))  * 1e4, 2),
            'max_bps':     round(float(np.max(arr))  * 1e4, 2),
        })
    return rows


# ── Full response payload ──────────────────────────────────────────────────────
def _build(tab: str = '3d') -> dict:
    metrics = compute_metrics(_sim.orderbook)
    bids, asks, _ = _sim.orderbook.get_state()
    regime = (_sim.orderbook_history[-1].get('regime', 'normal')
              if _sim.orderbook_history else 'normal')

    l2_bids = sorted(
        [{'price': float(p), 'volume': int(v)}
         for p, v in zip(bids.index[:20], bids.values[:20])],
        key=lambda x: -x['price'])
    l2_asks = sorted(
        [{'price': float(p), 'volume': int(v)}
         for p, v in zip(asks.index[:20], asks.values[:20])],
        key=lambda x: x['price'])

    trades = [
        {'time': t.time, 'side': t.side,
         'price': round(t.price, 4), 'size': t.size}
        for t in list(_sim.orderbook.trade_history)[-40:][::-1]
    ]

    rh    = list(_sim.regime_model.history)
    total = max(1, len(rh))
    regime_dist = {r: {'count': rh.count(r),
                       'pct':   round(rh.count(r) / total * 100, 1)}
                   for r in ['low', 'normal', 'high']}

    return {
        'status': {
            'step':         _sim.t,
            'regime':       regime,
            'trades_count': len(_sim.orderbook.trade_history),
            'total_volume': _sim.orderbook.total_volume,
            'mm_inventory': _sim.market_maker.inventory,
            'mm_pnl':       round(_sim.market_maker.mark_to_market, 2),
            'mm_cash':      round(_sim.market_maker.cash, 2),
            'best_bid':     _sim.orderbook.get_best_bid(),
            'best_ask':     _sim.orderbook.get_best_ask(),
            'mid':          _safe(metrics.get('Mid Price')),
            'spread_bps':   _safe(metrics.get('Spread (bps)')),
            'vwap':         _safe(metrics.get('VWAP')),
        },
        'metrics':       _safe_dict(metrics),
        'charts':        _charts_for_tab(tab),
        'l2':            {'bids': l2_bids, 'asks': l2_asks},
        'trades':        trades,
        'regime_dist':   regime_dist,
        'rolling_stats': _rolling_stats() if tab == 'stats' else [],
        'scenarios':     list(SCENARIOS.keys()),
        'active_tab':    tab,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/state')
def api_state():
    return jsonify(_build(request.args.get('tab', '3d')))

@app.route('/api/step', methods=['POST'])
def api_step():
    body = request.get_json(silent=True) or {}
    n    = int(min(body.get('n', 1), 500))
    tab  = body.get('tab', '3d')
    for _ in range(n):
        _sim.step()
    return jsonify(_build(tab))

@app.route('/api/reset', methods=['POST'])
def api_reset():
    tab = (request.get_json(silent=True) or {}).get('tab', '3d')
    _sim.reset()
    return jsonify(_build(tab))

@app.route('/api/scenario', methods=['POST'])
def api_scenario():
    global _sim
    body = request.get_json(silent=True) or {}
    name = body.get('scenario', '🟢  Normal Market')
    tab  = body.get('tab', '3d')
    if name in SCENARIOS:
        _sim = OrderBookSimulation(scenario=name)
    return jsonify(_build(tab))

@app.route('/api/inject', methods=['POST'])
def api_inject():
    body  = request.get_json(silent=True) or {}
    tab   = body.get('tab', '3d')
    order = {
        'type': body.get('type', 'market'),
        'side': body.get('side', 'buy'),
        'size': int(body.get('size', 100)),
    }
    if body.get('price'):
        order['price'] = float(body['price'])
    _sim.orderbook.step(order)
    _sim._record_state()
    return jsonify(_build(tab))

@app.route('/api/export')
def api_export():
    df  = pd.DataFrame([{
        'time': t.time, 'side': t.side,
        'price': t.price, 'size': t.size,
    } for t in list(_sim.orderbook.trade_history)])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    bio = io.BytesIO(buf.getvalue().encode('utf-8'))
    bio.seek(0)
    return send_file(bio, mimetype='text/csv',
                     as_attachment=True, download_name='trade_history.csv')


if __name__ == '__main__':
    print('\n  ┌─────────────────────────────────────────┐')
    print('  │  3D Order Book Liquidity Engine          │')
    print('  │  http://localhost:5000                   │')
    print('  └─────────────────────────────────────────┘\n')
    app.run(debug=False, port=5000, threaded=True)