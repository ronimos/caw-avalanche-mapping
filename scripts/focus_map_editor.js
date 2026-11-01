// Model-location editor for focus_stations_map.html (built by focus_map.py).
//
// Click a WMP point, a station, or anywhere on the map → the nearest SNOWPACK
// station opens in the side panel, where it can be marked Add / Remove for a
// set of aspects.  Save posts the edits to save_model_locations.php, which
// keeps one shared model_locations.json (with a timestamped backup of the
// previous version) for scripts/sync_pro_files.py --selection to apply on top
// of the baseline (nearest station to every WMP point, all aspects).  Unsaved
// edits are kept as a draft in this browser.  Without the PHP endpoint (e.g.
// opened as a local file) Download / Import still work.
//
// Data comes from window.FOCUS_EDITOR_DATA, written by focus_map.py.
window.addEventListener('load', function () {
  'use strict';
  const D = window.FOCUS_EDITOR_DATA;
  const map = window[D.map];
  const layer = k => window[D.layers[k]];
  const ASPECTS = ['Flat', 'N', 'E', 'S', 'W'];
  const STORE_KEY = 'caw-model-locations-' + D.season;

  // ── station lookup ────────────────────────────────────────────────────────
  const stations = {};                       // id → {id, lat, lon, elev}
  D.stations.forEach(([id, lat, lon, elev]) => { stations[id] = {id, lat, lon, elev}; });
  const baseline = D.baseline;               // id → [point names]
  const hwy = Object.fromEntries(D.hwy);     // id → km from road

  function distKm(lat1, lon1, lat2, lon2) {
    const r = Math.PI / 180, a = Math.sin((lat2 - lat1) * r / 2) ** 2
      + Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin((lon2 - lon1) * r / 2) ** 2;
    return 12742 * Math.asin(Math.sqrt(a));
  }
  function nearest(lat, lon) {
    let best = null, bd = Infinity;
    for (const s of Object.values(stations)) {
      const d = distKm(lat, lon, s.lat, s.lon);
      if (d < bd) { bd = d; best = s; }
    }
    return [best, bd];
  }

  // ── selections: `sel` = working copy, `saved` = server copy ──────────────
  // saved is null until the server answers (or when there is no endpoint).
  let sel = {}, saved = null, savedAt = null;
  let draft = null;                           // {sel, ts} left from an earlier visit
  try { draft = JSON.parse(localStorage.getItem(STORE_KEY) || 'null'); } catch (e) { draft = null; }
  if (draft && !draft.sel) draft = {sel: draft, ts: null};   // pre-save-button format
  if (draft) sel = draft.sel;
  function persist() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify({sel, ts: new Date().toISOString()})); }
    catch (e) { /* private mode */ }
  }
  const key = r => r ? `${r.action}|${r.aspects.join(',')}|${r.note || ''}` : '';
  const isSaved = id => saved !== null && key(sel[id]) === key(saved[id]);
  function nUnsaved() {
    if (saved === null) return Object.keys(sel).length;
    return [...new Set([...Object.keys(sel), ...Object.keys(saved)])].filter(id => !isSaved(id)).length;
  }
  function effectiveAspects(id) {
    const set = new Set(baseline[id] ? ASPECTS : []);
    const s = sel[id];
    if (s) s.aspects.forEach(a => s.action === 'add' ? set.add(a) : set.delete(a));
    return ASPECTS.filter(a => set.has(a));
  }

  // ── styles ────────────────────────────────────────────────────────────────
  const css = document.createElement('style');
  css.textContent = `
    #mle { position:fixed; top:10px; left:54px; z-index:1000; width:300px; max-height:calc(100vh - 20px);
      overflow:auto; background:#fff; color:#222; font:13px/1.4 system-ui,sans-serif;
      border-radius:8px; box-shadow:0 2px 10px rgba(0,0,0,.3); }
    #mle header { padding:8px 10px; background:#263238; color:#fff; border-radius:8px 8px 0 0;
      display:flex; justify-content:space-between; align-items:center; cursor:pointer; }
    #mle section { padding:8px 10px; border-top:1px solid #e0e0e0; }
    #mle h4 { margin:0 0 4px; font-size:13px; }
    #mle .muted { color:#666; font-size:12px; }
    #mle .row { display:flex; gap:4px; flex-wrap:wrap; margin:4px 0; }
    #mle button { font:inherit; font-size:12px; padding:3px 8px; border:1px solid #999; border-radius:4px;
      background:#f5f5f5; color:#222; cursor:pointer; }
    #mle button.on { background:#263238; color:#fff; border-color:#263238; }
    #mle button.add.on { background:#1b7837; border-color:#1b7837; }
    #mle button.remove.on { background:#b2182b; border-color:#b2182b; }
    #mle button.primary { background:#1565c0; color:#fff; border-color:#1565c0; }
    #mle input[type=text] { width:100%; box-sizing:border-box; font:inherit; padding:3px 5px; }
    #mle ul { list-style:none; margin:4px 0 0; padding:0; max-height:220px; overflow:auto; }
    #mle li { padding:3px 4px; border-radius:4px; cursor:pointer; }
    #mle li:hover { background:#eef3f8; }
    #mle .tag-add { color:#1b7837; font-weight:600; } #mle .tag-remove { color:#b2182b; font-weight:600; }
    #mle.collapsed section { display:none; }
    .mle-mark { font:bold 11px system-ui,sans-serif; color:#fff; border-radius:9px; padding:1px 5px;
      white-space:nowrap; border:1.5px solid #fff; box-shadow:0 0 3px rgba(0,0,0,.6); }
    .mle-tri { width:0; height:0; border-left:8px solid transparent; border-right:8px solid transparent;
      border-bottom:14px solid #d7191c; filter:drop-shadow(0 0 1px #000); }
    .mle-diamond { width:12px; height:12px; transform:rotate(45deg); background:#1b7837;
      border:2px solid #fff; box-shadow:0 0 0 1.5px #1b7837, 0 0 4px rgba(0,0,0,.6); }
    .mle-x { width:16px; height:16px; color:#b2182b; font:900 18px/16px system-ui,sans-serif; text-align:center;
      text-shadow:0 0 2px #fff, 0 0 3px #fff; }
    .mle-pending .mle-diamond { background:#fff; border:2px dashed #1b7837; box-shadow:0 0 4px rgba(0,0,0,.6); }
    .mle-pending .mle-x { opacity:.6; }
    .mle-pending .mle-mark { opacity:.75; border-style:dashed; }
    #mle .status-ok { color:#1b7837; } #mle .status-dirty { color:#b26a00; font-weight:600; }
    #mle .legend { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:4px; }
    #mle .legend span { display:inline-flex; align-items:center; gap:4px; }
    @media (max-width:600px) { #mle { left:10px; right:10px; width:auto; top:auto; bottom:10px; max-height:55vh; } }
  `;
  document.head.appendChild(css);

  // ── panel ─────────────────────────────────────────────────────────────────
  const panel = document.createElement('div');
  panel.id = 'mle';
  panel.innerHTML = `
    <header><b>Model locations</b><span id="mle-count"></span></header>
    <section><div class="row" style="flex-wrap:nowrap">
      <input id="mle-goto" type="text" placeholder="Search: WMP name, station, place, or lat, lon">
      <button id="mle-go">Go</button></div></section>
    <section id="mle-edit"><span class="muted">Click a WMP point, a station, or anywhere on the
      map to edit the nearest SNOWPACK station.</span></section>
    <section><h4>Edits</h4>
      <div class="legend muted">
        <span><span class="mle-diamond" style="display:inline-block;transform:rotate(45deg) scale(.8)"></span> added</span>
        <span><span class="mle-x" style="display:inline-block">✕</span> removed</span>
        <span>dashed = not saved yet</span></div>
      <ul id="mle-list"></ul>
      <div id="mle-status" class="muted"></div>
      <div class="row">
        <button id="mle-save" class="primary" disabled>Save</button>
        <button id="mle-dl">Download JSON</button>
        <button id="mle-imp">Import…</button>
        <button id="mle-clear">Clear all</button>
        <input id="mle-file" type="file" accept=".json,application/json" hidden>
      </div>
    </section>`;
  document.body.appendChild(panel);
  L.DomEvent.disableClickPropagation(panel);
  L.DomEvent.disableScrollPropagation(panel);
  panel.querySelector('header').onclick = () => panel.classList.toggle('collapsed');
  const editBox = panel.querySelector('#mle-edit');

  function el(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => k === 'class' ? n.className = v : n[k] = v);
    kids.forEach(k => n.append(k));
    return n;
  }

  let tempLine = null;
  function openEditor(id, source) {
    const s = stations[id];
    if (!s) return;
    if (tempLine) { map.removeLayer(tempLine); tempLine = null; }
    if (source.lat !== undefined) {
      tempLine = L.polyline([[source.lat, source.lon], [s.lat, s.lon]],
        {color: '#1565c0', weight: 2, dashArray: '4 4', interactive: false}).addTo(map);
    }
    panel.classList.remove('collapsed');
    const cur = sel[id];
    let action = cur ? cur.action : (baseline[id] ? 'remove' : 'add');
    let aspects = new Set(cur ? cur.aspects : ASPECTS);

    const status = baseline[id]
      ? `In the baseline (WMP: ${baseline[id].join(', ')}) — all aspects`
      : 'Not in the baseline';
    const extra = [];
    if (hwy[id] !== undefined) extra.push(`${hwy[id].toFixed(1)} km from the Pamir Highway`);
    if (source.label) extra.push(source.label);

    const actBtns = ['add', 'remove'].map(a => el('button', {class: a, textContent: a === 'add' ? 'Add' : 'Remove',
      onclick: () => { action = a; refresh(); }}));
    const aspBtns = ASPECTS.map(a => el('button', {textContent: a,
      onclick: () => { aspects.has(a) ? aspects.delete(a) : aspects.add(a); refresh(); }}));
    const note = el('input', {type: 'text', placeholder: 'Note (optional)', value: cur ? cur.note || '' : ''});
    const result = el('div', {class: 'muted'});
    const save = el('button', {class: 'primary', textContent: 'Apply'});
    const undo = el('button', {textContent: 'Discard edit', disabled: !cur});

    function refresh() {
      actBtns.forEach(b => b.classList.toggle('on', b.classList.contains(action)));
      aspBtns.forEach(b => b.classList.toggle('on', aspects.has(b.textContent)));
      const after = new Set(baseline[id] ? ASPECTS : []);
      aspects.forEach(a => action === 'add' ? after.add(a) : after.delete(a));
      const list = ASPECTS.filter(a => after.has(a));
      result.textContent = 'After this edit the model runs: ' + (list.length ? list.join(', ') : 'nothing here');
      save.disabled = aspects.size === 0;
    }
    save.onclick = () => {
      sel[id] = {station: id, action, aspects: ASPECTS.filter(a => aspects.has(a)), note: note.value.trim(),
        source: source.text || '', lat: s.lat, lon: s.lon, elevation: s.elev,
        updated: new Date().toISOString()};
      persist(); render(); openEditor(id, source);
    };
    undo.onclick = () => { delete sel[id]; persist(); render(); openEditor(id, source); };

    editBox.replaceChildren(
      el('h4', {textContent: `Station ${id} · ${Math.round(s.elev)} m`}),
      el('div', {class: 'muted', textContent: status}),
      ...extra.map(t => el('div', {class: 'muted', textContent: t})),
      el('div', {class: 'row'}, ...actBtns),
      el('div', {class: 'muted', textContent: 'Aspects'}),
      el('div', {class: 'row'}, ...aspBtns),
      note, result,
      el('div', {class: 'row'}, save, undo));
    refresh();
  }

  // ── base layers (built here so every marker opens the editor) ──────────────
  function clickable(marker, id, source) {
    marker.on('click', ev => { L.DomEvent.stopPropagation(ev); openEditor(id, source); });
    return marker;
  }
  D.points.forEach(([name, lat, lon, id, km, colour]) => {
    const s = stations[id];
    L.polyline([[lat, lon], [s.lat, s.lon]], {color: colour, weight: 2, opacity: .8, dashArray: '5 5',
      interactive: false}).addTo(layer('points'));
    clickable(L.circleMarker([lat, lon], {radius: 7, color: '#fff', weight: 1.5, fillColor: colour, fillOpacity: .95})
      .bindTooltip(`${name} → ${id} (${km.toFixed(1)} km)`),
      id, {text: `WMP point ${name}`, label: `From WMP point ${name} (${km.toFixed(1)} km)`, lat, lon})
      .addTo(layer('points'));
  });
  Object.entries(D.colours).forEach(([id, colour]) => {
    const s = stations[id];
    clickable(L.marker([s.lat, s.lon], {icon: L.divIcon({className: '', iconSize: [14, 14], iconAnchor: [7, 7],
      html: `<div style="width:12px;height:12px;background:${colour};border:2px solid #000"></div>`})})
      .bindTooltip(`Station ${id} (${Math.round(s.elev)} m)`), id, {text: 'WMP station'})
      .addTo(layer('focus'));
  });
  D.hwy.forEach(([id, km]) => {
    const s = stations[id];
    clickable(L.marker([s.lat, s.lon], {icon: L.divIcon({className: '', iconSize: [16, 14], iconAnchor: [8, 9],
      html: '<div class="mle-tri"></div>'})})
      .bindTooltip(`Highway station ${id} (${Math.round(s.elev)} m, ${km.toFixed(1)} km from road)`),
      id, {text: 'Pamir Highway station'}).addTo(layer('hwy'));
  });
  Object.values(stations).forEach(s => {
    clickable(L.circleMarker([s.lat, s.lon], {radius: 3, color: '#37474f', weight: 1, fillColor: '#eceff1',
      fillOpacity: .9}).bindTooltip(`Station ${s.id} (${Math.round(s.elev)} m)`), s.id, {text: 'Station'})
      .addTo(layer('all'));
  });

  map.on('click', ev => {
    const [s, km] = nearest(ev.latlng.lat, ev.latlng.lng);
    if (!s) return;
    const at = `${ev.latlng.lat.toFixed(4)}, ${ev.latlng.lng.toFixed(4)}`;
    openEditor(s.id, {text: `Map click ${at}`, label: `Nearest to ${at} (${km.toFixed(1)} km)`,
      lat: ev.latlng.lat, lon: ev.latlng.lng});
  });

  // ── edits layer + list ────────────────────────────────────────────────────
  function render() {
    const lyr = layer('edits');
    lyr.clearLayers();
    const list = panel.querySelector('#mle-list');
    list.replaceChildren();
    const edits = Object.values(sel).sort((a, b) => a.station.localeCompare(b.station, undefined, {numeric: true}));
    edits.forEach(e => {
      const s = stations[e.station] || e;
      const sign = e.action === 'add' ? '+' : '−';
      const done = isSaved(e.station);
      const shape = e.action === 'add' ? '<div class="mle-diamond"></div>' : '<div class="mle-x">✕</div>';
      // Diamond / ✕ on the station itself, aspect label beside it.
      clickable(L.marker([s.lat, s.lon], {zIndexOffset: 1000, icon: L.divIcon({
        className: done ? '' : 'mle-pending', iconSize: [16, 16], iconAnchor: [8, 8],
        html: `<div style="position:relative">${shape}<span class="mle-mark" style="position:absolute;left:18px;top:-1px;`
          + `background:${e.action === 'add' ? '#1b7837' : '#b2182b'}">${sign} ${e.aspects.join(' ')}</span></div>`})})
        .bindTooltip(`Station ${e.station}: ${e.action} ${e.aspects.join(', ')}${done ? '' : ' (not saved yet)'}`),
        e.station, {text: e.source}).addTo(lyr);
      const li = el('li', {}, el('span', {class: 'tag-' + e.action, textContent: `${sign} ${e.station}`}),
        ` ${e.aspects.join(', ')}` + (e.note ? ` — ${e.note}` : '') + (done ? '' : ' •'));
      li.onclick = () => { map.setView([s.lat, s.lon], Math.max(map.getZoom(), 10)); openEditor(e.station, {text: e.source}); };
      list.append(li);
    });
    if (!edits.length) list.append(el('li', {class: 'muted', textContent: 'No edits yet'}));
    let nSt = 0, nFiles = 0;
    new Set([...Object.keys(baseline), ...Object.keys(sel)]).forEach(id => {
      const n = effectiveAspects(id).length;
      if (n) { nSt++; nFiles += n; }
    });
    panel.querySelector('#mle-count').textContent = `${nSt} stations · ${nFiles} files`;
    updateStatus();
  }

  // ── save / load (server) ──────────────────────────────────────────────────
  const statusEl = panel.querySelector('#mle-status');
  const saveBtn = panel.querySelector('#mle-save');
  let serverMsg = 'Connecting to the save service…';
  function updateStatus() {
    const n = nUnsaved();
    if (saved === null) {
      statusEl.className = 'muted';
      statusEl.textContent = serverMsg;
      saveBtn.disabled = true;
      return;
    }
    statusEl.className = n ? 'status-dirty' : 'status-ok';
    statusEl.textContent = n ? `${n} unsaved change(s) (•)`
      : 'All edits saved' + (savedAt ? ` · ${new Date(savedAt).toLocaleString()}` : '');
    saveBtn.disabled = n === 0;
  }
  const toMap = rows => Object.fromEntries((rows || []).map(r => [String(r.station), r]));

  saveBtn.onclick = () => {
    saveBtn.disabled = true;
    statusEl.textContent = 'Saving…';
    fetch(D.save_url, {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(exportDoc())})
      .then(r => r.json().then(j => { if (!r.ok) throw new Error(j.error || r.statusText); return j; }))
      .then(j => {
        saved = JSON.parse(JSON.stringify(sel));
        if (j.saved) savedAt = j.saved;
        persist(); render();
        if (j.backup) statusEl.textContent += ` · previous version backed up as ${j.backup}`;
      })
      .catch(err => { alert('Save failed: ' + err.message + '\nYour edits are still kept in this browser.'); updateStatus(); });
  };

  function loadServer() {
    if (!D.save_url || location.protocol === 'file:') {
      serverMsg = 'Saving needs the web server — use Download / Import here.';
      return updateStatus();
    }
    fetch(D.save_url, {cache: 'no-store'})
      .then(r => r.status === 404 ? {selections: []} : r.ok ? r.json() : Promise.reject(new Error(r.statusText)))
      .then(doc => {
        saved = toMap(doc.selections);
        savedAt = doc.saved || null;
        const differs = draft && Object.keys(draft.sel).length
          && JSON.stringify(Object.keys(sel).sort().map(id => key(sel[id])))
             !== JSON.stringify(Object.keys(saved).sort().map(id => key(saved[id])));
        if (!(differs && confirm('This browser has unsaved edits'
            + (draft.ts ? ` from ${new Date(draft.ts).toLocaleString()}` : '')
            + ' that differ from the saved file.\n\nOK = keep working on them\nCancel = load the saved file instead'))) {
          sel = JSON.parse(JSON.stringify(saved));
          persist();
        }
        render();
      })
      .catch(err => { serverMsg = `Save service unavailable (${err.message}) — use Download / Import.`; updateStatus(); });
  }
  window.addEventListener('beforeunload', ev => {
    if (saved !== null && nUnsaved()) { ev.preventDefault(); ev.returnValue = ''; }
  });

  // ── search: lat, lon · WMP point name · station id · place name ──────────
  // Local names first (with suggestions); anything else goes to OpenStreetMap
  // Nominatim, the same service as the magnifier control.
  const searchIn = panel.querySelector('#mle-goto');
  const suggest = el('datalist', {id: 'mle-suggest'});
  D.points.forEach(([name]) => suggest.append(el('option', {value: name})));
  Object.keys(baseline).concat(D.hwy.map(h => h[0])).forEach(id => suggest.append(el('option', {value: id})));
  panel.append(suggest);
  searchIn.setAttribute('list', 'mle-suggest');

  let gotoMarker = null;
  function goTo(lat, lon, what) {
    if (gotoMarker) map.removeLayer(gotoMarker);
    gotoMarker = L.marker([lat, lon]).bindTooltip(what).addTo(map);
    map.setView([lat, lon], Math.max(map.getZoom(), 11));
    const [s, km] = nearest(lat, lon);
    if (s) openEditor(s.id, {text: what, label: `Nearest to ${what} (${km.toFixed(1)} km)`, lat, lon});
  }
  function search() {
    const q = searchIn.value.trim();
    if (!q) return;
    const ll = q.match(/^(-?\d+(?:\.\d+)?)\s*[,;\s]\s*(-?\d+(?:\.\d+)?)$/);
    if (ll && Math.abs(+ll[1]) <= 90 && Math.abs(+ll[2]) <= 180) return goTo(+ll[1], +ll[2], `${+ll[1]}, ${+ll[2]}`);
    if (stations[q]) {
      const s = stations[q];
      map.setView([s.lat, s.lon], Math.max(map.getZoom(), 11));
      return openEditor(q, {text: `Station ${q}`});
    }
    const lq = q.toLowerCase();
    const pts = D.points.filter(p => p[0].toLowerCase() === lq);
    const hits = pts.length ? pts : D.points.filter(p => p[0].toLowerCase().includes(lq));
    if (hits.length === 1) {
      const [name, lat, lon, id, km] = hits[0];
      map.setView([lat, lon], Math.max(map.getZoom(), 11));
      return openEditor(id, {text: `WMP point ${name}`, label: `From WMP point ${name} (${km.toFixed(1)} km)`, lat, lon});
    }
    if (hits.length > 1) {
      map.fitBounds(L.latLngBounds(hits.map(p => [p[1], p[2]])).pad(0.3));
      return alert(`${hits.length} WMP points match "${q}": ${hits.map(p => p[0]).join(', ')}`);
    }
    fetch('https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q=' + encodeURIComponent(q))
      .then(r => r.json())
      .then(res => res.length ? goTo(+res[0].lat, +res[0].lon, res[0].display_name.split(',').slice(0, 2).join(','))
                              : alert(`Nothing found for "${q}"`))
      .catch(err => alert('Place search failed: ' + err.message));
  }
  panel.querySelector('#mle-go').onclick = search;
  searchIn.onkeydown = ev => { if (ev.key === 'Enter') search(); };

  // ── export / import ───────────────────────────────────────────────────────
  function exportDoc() {
    return {schema: 'caw-model-locations/1', season: D.season, run_dir: D.run_dir,
      exported: new Date().toISOString(),
      baseline: 'nearest SNOWPACK station to each WMP point in data/FOCUS_Ava.kml, all aspects',
      aspects: {Flat: '', N: '1', E: '2', S: '3', W: '4'},
      selections: Object.values(sel)};
  }
  panel.querySelector('#mle-dl').onclick = () => {
    const doc = exportDoc();
    const a = el('a', {href: URL.createObjectURL(new Blob([JSON.stringify(doc, null, 2)], {type: 'application/json'})),
      download: 'model_locations.json'});
    document.body.append(a); a.click(); a.remove();
  };
  const fileIn = panel.querySelector('#mle-file');
  panel.querySelector('#mle-imp').onclick = () => fileIn.click();
  fileIn.onchange = () => {
    const f = fileIn.files[0];
    if (!f) return;
    f.text().then(txt => {
      const doc = JSON.parse(txt);
      const rows = (doc.selections || []).filter(r => stations[r.station] && ['add', 'remove'].includes(r.action)
        && Array.isArray(r.aspects) && r.aspects.every(a => ASPECTS.includes(a)));
      if (Object.keys(sel).length && !confirm(`Replace your ${Object.keys(sel).length} current edit(s) with ${rows.length} from the file?`)) return;
      sel = Object.fromEntries(rows.map(r => [r.station, r]));
      persist(); render();
    }).catch(err => alert('Could not read file: ' + err.message)).finally(() => { fileIn.value = ''; });
  };
  panel.querySelector('#mle-clear').onclick = () => {
    if (Object.keys(sel).length && confirm('Clear all edits? Nothing changes on the server until you press Save '
        + '(and Save keeps a backup of the previous file).')) {
      sel = {}; persist(); render();
    }
  };

  render();
  loadServer();
});
