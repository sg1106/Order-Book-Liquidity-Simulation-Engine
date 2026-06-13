/* ════════════════════════════════════════════════════════════════════════
   script.js
   Frontend logic for the 3D Order Book Liquidity Engine
   Talks to the Flask backend (server.py) via the /api/* REST endpoints.
   ════════════════════════════════════════════════════════════════════════ */
   'use strict';

   // ── State ───────────────────────────────────────────────────────────────────
   const S = {
     tab:     '3d',
     running: false,
     timer:   null,
     speed:   300,
     data:    null,
   };
   
   // ── Helpers ─────────────────────────────────────────────────────────────────
   const $ = id => document.getElementById(id);
   
   const fmt = (v, dec = 4, pre = '', suf = '') =>
     v == null ? '—' : `${pre}${Number(v).toLocaleString('en-US',
       {minimumFractionDigits: dec, maximumFractionDigits: dec})}${suf}`;
   
   function post(url, body = {}) {
     return fetch(url, {
       method: 'POST',
       headers: {'Content-Type': 'application/json'},
       body: JSON.stringify({...body, tab: S.tab}),
     }).then(r => r.json());
   }
   
   function plotly(id, fig, height) {
     if (!fig) return;
     const layout = {
       ...fig.layout,
       hoverlabel: {
         bgcolor: '#111820',
         bordercolor: '#1e2a35',
         font: {color: '#c8d8e8', family: 'JetBrains Mono, monospace', size: 11},
         ...(fig.layout && fig.layout.hoverlabel),
       },
     };
     if (height) layout.height = height;
   
     const div = document.getElementById(id);
     if (!div) return;
   
     Plotly.react(div, fig.data, layout, {responsive: true, displayModeBar: false});
   
     // Plotly.react() can measure a 0x0 container if the tab panel was just
     // switched from display:none -> block in this same tick (the browser
     // hasn't reflowed yet), producing a blank/broken chart that only
     // "fixes itself" once something else (e.g. the Auto-run loop) triggers
     // another render after layout has settled. Force a resize on the next
     // animation frame so it renders correctly immediately.
     requestAnimationFrame(() => {
       try { Plotly.Plots.resize(div); } catch (e) { /* chart not ready yet */ }
     });
   }
   
   // ── Init ────────────────────────────────────────────────────────────────────
   async function init() {
     const data = await fetch(`/api/state?tab=${S.tab}`).then(r => r.json());
   
     // Populate scenario dropdown
     const sel = $('sb-scenario');
     data.scenarios.forEach(s => {
       const opt = document.createElement('option');
       opt.value = s;
       opt.textContent = s;
       if (s.includes('Normal')) opt.selected = true;
       sel.appendChild(opt);
     });
     sel.onchange = () => setScenario(sel.value);
   
     render(data);
     $('loading').classList.add('hidden');
   }
   
   // ── API actions ─────────────────────────────────────────────────────────────
   async function doStep(n = 1) {
     const data = await post('/api/step', {n});
     render(data);
   }
   
   async function resetSim() {
     stopAuto();
     const data = await post('/api/reset');
     render(data);
   }
   
   async function setScenario(name) {
     stopAuto();
     const data = await post('/api/scenario', {scenario: name});
     render(data);
   }
   
   async function injectOrder() {
     const type  = $('inj-type').value;
     const side  = $('inj-side').value;
     const size  = parseInt($('inj-size').value) || 100;
     const price = parseFloat($('inj-price').value) || null;
   
     const body = {type, side, size};
     if (price && type !== 'market') body.price = price;
   
     const data = await post('/api/inject', body);
     render(data);
   }
   
   function exportCSV() {
     window.open('/api/export', '_blank');
   }
   
   function togglePriceField() {
     const t = $('inj-type').value;
     $('price-row').style.display = (t === 'limit' || t === 'cancel') ? '' : 'none';
   }
   
   // ── Auto-run ────────────────────────────────────────────────────────────────
   function toggleAuto() {
     S.running ? stopAuto() : startAuto();
   }
   
   function startAuto() {
     if (S.running) return;
     S.running = true;
     const btn = $('btn-auto');
     btn.textContent = '⏹ Stop';
     btn.classList.add('stop');
     scheduleAuto();
   }
   
   function stopAuto() {
     S.running = false;
     clearTimeout(S.timer);
     const btn = $('btn-auto');
     btn.textContent = '▶▶ Auto';
     btn.classList.remove('stop');
   }
   
   function scheduleAuto() {
     S.speed = parseInt($('sb-speed').value) || 300;
     if (!S.running) return;
     S.timer = setTimeout(async () => {
       try {
         const d = await post('/api/step', {n: 1});
         render(d);
       } catch (e) { /* ignore transient network errors */ }
       scheduleAuto();
     }, S.speed);
   }
   
   // ── Tab switching ───────────────────────────────────────────────────────────
   async function switchTab(tab) {
     S.tab = tab;
     document.querySelectorAll('.tab-btn').forEach(b =>
       b.classList.toggle('active', b.dataset.tab === tab));
     document.querySelectorAll('.tab-panel').forEach(p =>
       p.classList.toggle('active', p.id === `panel-${tab}`));
   
     const data = await fetch(`/api/state?tab=${tab}`).then(r => r.json());
     render(data);
   }
   
   // ── Render dispatcher ──────────────────────────────────────────────────────
   function render(data) {
     S.data = data;
     renderHeader(data.status);
     renderCharts(data.charts, S.tab);
     renderMetrics(data.metrics);
   
     if (S.tab === 'analytics' || S.tab === 'l2') renderTrades(data.trades);
     if (S.tab === 'l2')    renderL2(data.l2, data.status);
     if (S.tab === 'mm')    renderMM(data.status, data.regime_dist);
     if (S.tab === 'stats') renderStats(data.rolling_stats);
   
     // Keep inject-price field synced to current mid
     if (data.status.mid) $('inj-price').value = data.status.mid.toFixed(1);
   }
   
   // ── Header ──────────────────────────────────────────────────────────────────
   function renderHeader(s) {
     const regimeDot   = {low: 'dot-low', normal: 'dot-normal', high: 'dot-high'};
     const regimeLabel = {low: 'LOW VOL', normal: 'NORMAL',  high: 'HIGH VOL'};
   
     const dot = $('regime-dot');
     dot.className = `dot ${regimeDot[s.regime] || 'dot-normal'}`;
     $('regime-label').textContent = regimeLabel[s.regime] || 'NORMAL';
   
     $('h-step').textContent   = s.step.toLocaleString();
     $('h-trades').textContent = s.trades_count.toLocaleString();
     $('h-vol').textContent    = s.total_volume.toLocaleString();
     $('h-spread').textContent = s.spread_bps != null ? `${s.spread_bps.toFixed(1)} bps` : '—';
     $('h-vwap').textContent   = s.vwap != null ? `$${s.vwap.toFixed(3)}` : '—';
     $('h-mid').textContent    = s.mid  != null ? `$${s.mid.toFixed(3)}`  : '—';
   
     const pnl   = s.mm_pnl;
     const pnlEl = $('h-pnl');
     pnlEl.textContent = pnl != null
       ? pnl.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})
       : '—';
     pnlEl.style.color = pnl >= 0 ? 'var(--bid)' : 'var(--ask)';
   
     // MM inventory badge
     const inv   = s.mm_inventory;
     const invEl = $('mm-inv-val');
     invEl.textContent = (inv >= 0 ? '+' : '') + inv.toLocaleString();
     invEl.style.color = inv > 0 ? 'var(--bid)' : (inv < 0 ? 'var(--ask)' : 'var(--dim)');
   }
   
   // ── Charts ──────────────────────────────────────────────────────────────────
   const TAB_CHARTS = {
     '3d':        [{id: 'chart-3d',           key: '3d',           h: 520}],
     'depth':     [{id: 'chart-depth',        key: 'depth',        h: 380},
                   {id: 'chart-slippage',     key: 'slippage',     h: 360}],
     'heatmap':   [{id: 'chart-heatmap',      key: 'heatmap',      h: 400},
                   {id: 'chart-ofi',          key: 'ofi',          h: 220},
                   {id: 'chart-vprofile',     key: 'vprofile',     h: 460}],
     'analytics': [{id: 'chart-regime-price', key: 'regime_price', h: 360},
                   {id: 'chart-tape',         key: 'tape',         h: 350},
                   {id: 'chart-cum-delta',    key: 'cum_delta',    h: 240}],
     'mm':        [{id: 'chart-mm',           key: 'mm',           h: 420}],
     'stats':     [{id: 'chart-returns',      key: 'returns',      h: 340},
                   {id: 'chart-price-hist',   key: 'price_hist',   h: 400}],
   };
   
   function renderCharts(charts, tab) {
     const defs = TAB_CHARTS[tab] || [];
     defs.forEach(({id, key, h}) => {
       if (charts[key]) plotly(id, charts[key], h);
     });
   }
   
   // ── Metrics panel ───────────────────────────────────────────────────────────
   function card(label, value, delta = '', deltaClass = '') {
     return `<div class="m-card">
       <div class="m-label">${label}</div>
       <div class="m-value">${value}</div>
       ${delta ? `<div class="m-delta ${deltaClass}">${delta}</div>` : ''}
     </div>`;
   }
   
   function twoCards(l1, v1, l2, v2) {
     return `<div class="m-row">${card(l1, v1)}${card(l2, v2)}</div>`;
   }
   
   function renderMetrics(m) {
     const ofi    = m['OFI'] || 0;
     const ofiLbl = ofi > 0.1 ? '↑ Buy pressure' : (ofi < -0.1 ? '↓ Sell pressure' : 'Balanced');
     const ofiCls = ofi > 0.1 ? 'up' : (ofi < -0.1 ? 'down' : '');
   
     const ac    = m['Autocorr'];
     const acLbl = ac != null
       ? (ac < -0.05 ? 'Mean-reverting' : (ac > 0.05 ? 'Trending' : 'Random walk'))
       : '';
   
     const cd    = m['Cum Delta'] || 0;
     const cdLbl = `${cd >= 0 ? 'Net buy' : 'Net sell'} ${Math.abs(Math.round(cd)).toLocaleString()}`;
     const cdVal = cd >= 0 ? `+${Math.round(cd).toLocaleString()}` : Math.round(cd).toLocaleString();
   
     const rv20   = m['RV-20'];
     const sp_bps = m['Spread (bps)'];
   
     const html = `
       <div class="mp-title">📊 Live Metrics</div>
       ${card('Mid Price', fmt(m['Mid Price'], 3, '$'))}
       ${twoCards('Best Bid', fmt(m['Best Bid'], 3, '$'), 'Best Ask', fmt(m['Best Ask'], 3, '$'))}
       ${card('VWAP', fmt(m['VWAP'], 3, '$'))}
       <hr>
       ${card('Spread', fmt(m['Spread'], 4), sp_bps != null ? `${fmt(sp_bps, 1)} bps` : '')}
       <hr>
       ${card('Book OFI',        fmt(m['OFI'], 3), ofiLbl, ofiCls)}
       ${card('Top-of-Book Imb', fmt(m['Top Imbalance'], 3))}
       ${card('Cum Delta', cdVal, cdLbl)}
       <hr>
       ${card('Depth ±5t',  fmt(m['Depth (±5t)'], 0, '', '  u'))}
       ${card('Depth ±10t', fmt(m['Depth (±10t)'], 0, '', '  u'))}
       ${card('Slip ×100',   fmt(m['Slip (100)'], 4))}
       ${card('Slip ×500',   fmt(m['Slip (500)'], 4))}
       ${card('Slip ×1000',  fmt(m['Slip (1000)'], 4))}
       <hr>
       ${card('Realized Vol', fmt(m['Realized Vol'], 4), rv20 != null ? `RV-20: ${fmt(rv20, 4)}` : '')}
       ${card("Kyle's λ", fmt(m["Kyle's λ"], 6))}
       ${card('Amihud',   fmt(m['Amihud'], 8))}
       ${card('Return Autocorr', fmt(m['Autocorr'], 3), acLbl)}
       ${card('Sharpe', fmt(m['Sharpe'], 3))}
       <hr>
       ${card('Traded Vol', m['Total Volume'] != null ? Math.round(m['Total Volume']).toLocaleString() : '—')}
       ${twoCards(
         'Book Bid Vol', m['Total Bid Vol'] != null ? Math.round(m['Total Bid Vol']).toLocaleString() : '—',
         'Book Ask Vol', m['Total Ask Vol'] != null ? Math.round(m['Total Ask Vol']).toLocaleString() : '—'
       )}
     `;
   
     ['mp-3d', 'mp-depth'].forEach(id => {
       const el = $(id);
       if (el) el.innerHTML = html;
     });
   }
   
   // ── Trades table ────────────────────────────────────────────────────────────
   function renderTrades(trades) {
     const tb = $('trades-tbody');
     if (!tb) return;
     tb.innerHTML = trades.slice(0, 40).map(t => `
       <tr>
         <td>${t.time}</td>
         <td class="${t.side === 'buy' ? 'buy-side' : 'sell-side'}">${t.side === 'buy' ? '▲ BUY' : '▼ SELL'}</td>
         <td>${fmt(t.price, 4)}</td>
         <td>${t.size.toLocaleString()}</td>
       </tr>`).join('');
   }
   
   // ── L2 book ─────────────────────────────────────────────────────────────────
   function renderL2(l2, status) {
     const maxBid = Math.max(...l2.bids.map(r => r.volume), 1);
     const maxAsk = Math.max(...l2.asks.map(r => r.volume), 1);
   
     function rows(data, isBid, maxV) {
       return data.map(r => {
         const pct   = Math.round(r.volume / maxV * 100);
         const color = isBid ? 'var(--bid)' : 'var(--ask)';
         const bar   = `<div class="depth-bar" style="width:${pct * 0.8}px;background:${color};opacity:0.4"></div>`;
         return `<tr>
           <td style="color:${color};font-weight:600">${r.price.toFixed(3)}</td>
           <td>${r.volume.toLocaleString()}</td>
           <td>${bar}</td>
         </tr>`;
       }).join('');
     }
   
     const bTb = $('l2-bids-body'); if (bTb) bTb.innerHTML = rows(l2.bids, true,  maxBid);
     const aTb = $('l2-asks-body'); if (aTb) aTb.innerHTML = rows(l2.asks, false, maxAsk);
   
     const midEl = $('l2-mid-bar');
     if (midEl && status) {
       const bb = status.best_bid ? status.best_bid.toFixed(3) : '—';
       const ba = status.best_ask ? status.best_ask.toFixed(3) : '—';
       const mp = status.mid      ? status.mid.toFixed(3)      : '—';
       const sp = (status.best_bid != null && status.best_ask != null)
         ? (status.best_ask - status.best_bid).toFixed(3)
         : '—';
   
       midEl.innerHTML = `
         <span class="lbl">BEST BID</span><span class="bid-val">${bb}</span>
         <span class="sep">|</span>
         <span class="lbl">MID</span><span class="mid-val">${mp}</span>
         <span class="sep">|</span>
         <span class="lbl">BEST ASK</span><span class="ask-val">${ba}</span>
         <span class="sep">&nbsp;&nbsp;</span>
         <span class="lbl">SPREAD</span><span style="font-size:0.9rem">${sp}</span>`;
     }
   }
   
   // ── Market Maker tab ────────────────────────────────────────────────────────
   function renderMM(s, rDist) {
     const cards = $('mm-summary-cards');
     if (cards) {
       const pnlColor = s.mm_pnl >= 0 ? 'var(--bid)' : 'var(--ask)';
       const invColor = s.mm_inventory > 0
         ? 'var(--bid)'
         : (s.mm_inventory < 0 ? 'var(--ask)' : 'var(--dim)');
   
       cards.innerHTML = `
         <div class="mm-card">
           <div class="mm-card-label">Mark-to-Market</div>
           <div class="mm-card-value" style="color:${pnlColor}">
             ${s.mm_pnl.toLocaleString('en-US', {minimumFractionDigits: 2})}
           </div>
         </div>
         <div class="mm-card">
           <div class="mm-card-label">Cash</div>
           <div class="mm-card-value">
             ${s.mm_cash.toLocaleString('en-US', {minimumFractionDigits: 2})}
           </div>
         </div>
         <div class="mm-card">
           <div class="mm-card-label">Inventory</div>
           <div class="mm-card-value" style="color:${invColor}">
             ${s.mm_inventory > 0 ? '+' : ''}${s.mm_inventory.toLocaleString()}
           </div>
         </div>`;
     }
   
     const rc = $('regime-cards');
     if (rc && rDist) {
       rc.innerHTML = [
         {label: '🔵 Low Vol',  key: 'low',    color: '#5baeff'},
         {label: '🟢 Normal',   key: 'normal', color: '#00d4a0'},
         {label: '🔴 High Vol', key: 'high',   color: '#ff5555'},
       ].map(r => `
         <div class="regime-card">
           <div class="regime-card-label">${r.label}</div>
           <div class="regime-card-count" style="color:${r.color}">${rDist[r.key].count.toLocaleString()}</div>
           <div class="regime-card-pct">${rDist[r.key].pct}%</div>
         </div>`).join('');
     }
   }
   
   // ── Rolling stats ───────────────────────────────────────────────────────────
   function renderStats(rows) {
     const tb = $('stats-tbody');
     if (!tb) return;
   
     if (!rows || !rows.length) {
       tb.innerHTML = '<tr><td colspan="7" style="color:var(--dim);text-align:center">Step more to accumulate data…</td></tr>';
       return;
     }
   
     tb.innerHTML = rows.map(r => `
       <tr>
         <td>${r.window}</td>
         <td>${r.mean_bps}</td>
         <td>${r.std_bps}</td>
         <td>${r.ann_vol_pct}%</td>
         <td>${r.autocorr != null ? r.autocorr : '—'}</td>
         <td>${r.min_bps}</td>
         <td>${r.max_bps}</td>
       </tr>`).join('');
   }
   
   // ── Boot ────────────────────────────────────────────────────────────────────
   window.addEventListener('DOMContentLoaded', init);