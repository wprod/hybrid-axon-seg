/* app.js -- Nerve Segmentation Validator frontend */

const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

class App {
  constructor() {
    this.stems = [];
    this.cur = null;          // current stem
    this.mode = 'navigate';

    // canvas state
    this.cvs = $('#canvas');
    this.ctx = this.cvs.getContext('2d');
    this.img = null;
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;

    // interaction
    this.dragging = false;
    this.lastM = null;
    this.spaceHeld = false;
    this.drawing = false;
    this.drawPts = [];

    // fiber mode (two-step lasso: outer then inner)
    this.fiberStep = 0;
    this.outerPoly = [];

    // fascicle polygon mode (click-to-add-point)
    this.fascPts = [];
    this.fascMouse = null;
    this._lastFascClick = 0;
    this.savedFasc = null;

    // exclusion polygon mode (click-to-add-point)
    this.exclPts = [];
    this.exclMouse = null;
    this._lastExclClick = 0;
    this.savedExcl = null;

    // pointer tracking (touch/pen/mouse unification)
    this._pointers = new Map();   // pointerId → {x, y, type}
    this._pinchDist = 0;
    this._pinchMid = null;

    // GT mode
    this.gtMode = false;

    // processed state per stem
    this.processedMap = new Map();

    // annotations (brief markers before overlay refreshes)
    this.anns = [];
    this.editCount = 0;

    // raw image for opacity blending
    this.rawImg = null;
    this.overlayOpacity = 0.85;

    // Operation queue — batches rapid edits into a single overlay refresh
    this._qItems   = [];    // pending {fn, id} objects
    this._qRunning = false;
    this._qDirty   = false; // overlay needs refresh after drain
    this._qTotal   = 0;     // ops added in current batch
    this._qDone    = 0;     // ops completed in current batch
    this._qLog     = [];    // [{id, label, state}] for action stack UI
    this._qLogSeq  = 0;     // unique id counter

    // Multi-user presence
    this._clientId = Math.random().toString(36).slice(2, 10);
    this._presenceTimer = null;
    this._presenceCounts = {};  // stem → viewer count

    // Touch-confirm for delete mode
    this._delConfirm = null;

    // Easter egg key buffer 💙
    this._keyBuffer = '';
    this._logoClicks = 0;
    this._logoTimer = null;

    this.boot();
  }

  async boot() {
    this.bindEvents();
    this.bindResize();
    // Load server config (gratio_map flag, etc.)
    try {
      const r = await fetch('/api/config');
      this.serverConfig = await r.json();
    } catch { this.serverConfig = {}; }
    // Hide G-ratio view button if disabled server-side
    if (!this.serverConfig.gratio_map) {
      const btn = $('[data-view="gratio_map"]');
      if (btn) btn.classList.add('hidden');
    }
    // Restore mode + stem from URL hash  (#gt:stem  or  #stem)
    const hash = location.hash ? decodeURIComponent(location.hash.slice(1)) : '';
    const isGt = hash.startsWith('gt:');
    const hashStem = isGt ? hash.slice(3) : hash;

    if (isGt) {
      this.gtMode = true;
      $('#btn-gt-toggle').classList.add('active');
      $('#gt-banner').classList.remove('hidden');
      $('#btn-recompute').classList.add('hidden');
      $('#btn-reset').classList.add('hidden');
      $('#btn-recompute-all').classList.add('hidden');
      $$('.view-btn').forEach(b => b.classList.add('hidden'));
      $('#btn-compare').classList.add('hidden');
      $('[data-mode="accept"]').classList.add('hidden');
      $('[data-mode="vessel"]').classList.remove('hidden');
      $('#gt-validate-panel').classList.remove('hidden');
      await this._loadGtList();
      if (hashStem) { await this._gtSelect(hashStem); $('#hint').classList.add('hidden'); }
    } else {
      await this.loadList();
      if (this.stems.length) {
        const initial = (hashStem && this.stems.includes(hashStem)) ? hashStem : this.stems[0];
        await this.select(initial);
        $('#hint').classList.add('hidden');
      }
    }
    // Presence: immediate poll + every 10s + re-ping when tab becomes visible again
    this._pollPresence();
    setInterval(() => this._pollPresence(), 10000);
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden && this.cur) { this._startPresence(this.cur); this._pollPresence(); }
    });
    window.addEventListener('pagehide', () => this._stopPresence());
    // Resume polling if batch was already running
    try {
      const r = await fetch('/api/recompute-all/status');
      const s = await r.json();
      if (s.running) {
        $('#btn-recompute-all').disabled = true;
        this._pollBatch();
      }
    } catch {}
  }

  /* ── Events ──────────────────────────────────────────────────────────── */

  bindEvents() {
    $$('.mode-btn').forEach(b => b.onclick = () => this.setMode(b.dataset.mode));
    $('#btn-gt-toggle').onclick = () => this.toggleGtMode();
    $('#btn-undo').onclick = () => this.undo();
    $('#btn-recompute').onclick = () => this.recompute();
    $('#btn-reset').onclick = () => this.reset();
    $('#btn-clear-fasc').onclick = () => this.clearFascicle();
    $('#btn-clear-excl').onclick = () => this.clearExclusion();
    $('#btn-recompute-all').onclick = () => this.recomputeAll();
    $('#btn-compare').onclick = () => window.open('/api/comparison', '_blank');
    $('#btn-prev').onclick = () => this.nav(-1);
    $('#btn-next').onclick = () => this.nav(1);

    $$('.view-btn').forEach(b => b.onclick = () => {
      if (this.cur) window.open(`/api/image/${this.cur}/${b.dataset.view}`, '_blank');
    });

    // Opacity slider
    $('#opacity-slider').oninput = e => {
      this.overlayOpacity = e.target.value / 100;
      $('#opacity-val').textContent = `${e.target.value}%`;
      this.render();
    };

    // Easter egg — triple-click on logo 💙
    $('.logo').onclick = () => {
      this._logoClicks++;
      clearTimeout(this._logoTimer);
      this._logoTimer = setTimeout(() => { this._logoClicks = 0; }, 600);
      if (this._logoClicks >= 3) {
        this._logoClicks = 0;
        this._marie('💙 Nerve Validator — fait avec amour pour toi, Marie !');
      }
    };

    // Pointer events — unified mouse / touch / stylus
    this.cvs.addEventListener('pointerdown', e => this.onDown(e));
    this.cvs.addEventListener('pointermove', e => this.onMove(e));
    document.addEventListener('pointerup', e => this.onUp(e));
    document.addEventListener('pointercancel', e => this.onUp(e));
    this.cvs.addEventListener('pointerleave', e => {
      this._pointers.delete(e.pointerId);
      if (this._pointers.size === 0) {
        if (this.dragging) { this.dragging = false; $('#viewer').classList.remove('panning'); }
        if (this.mode === 'fascicle' && this.fascMouse) { this.fascMouse = null; this.render(); }
        if (this.mode === 'exclude'  && this.exclMouse) { this.exclMouse = null; this.render(); }
      }
    });
    this.cvs.addEventListener('wheel', e => this.onWheel(e), { passive: false });
    this.cvs.addEventListener('contextmenu', e => e.preventDefault());

    document.addEventListener('keydown', e => this.onKey(e, true));
    document.addEventListener('keyup', e => this.onKey(e, false));
  }

  bindResize() {
    const v = $('#viewer');
    new ResizeObserver(() => {
      this.cvs.width = v.clientWidth;
      this.cvs.height = v.clientHeight;
      this.render();
    }).observe(v);
  }

  setMode(m) {
    if (this.mode === 'fiber' && m !== 'fiber') {
      this.fiberStep = 0; this.outerPoly = [];
    }
    if (this.drawing && m !== this.mode) {
      this.drawing = false; this.drawPts = [];
    }
    if (m !== 'fascicle') {
      this.fascPts = []; this.fascMouse = null;
    }
    if (m !== 'exclude') {
      this.exclPts = []; this.exclMouse = null;
    }
    this.mode = m;
    $$('.mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === m));
    $('#viewer').className = `mode-${m}${this.spaceHeld ? ' space-held' : ''}`;

    // Cursor hint label
    const hintEl = document.getElementById('cursor-hint');
    if (hintEl) {
      const hintLabels = {
        delete: 'Delete', fiber: '+Fiber', fascicle: 'Fascicle',
        'paint-axon': '+Axon', 'paint-outer': '+Myelin', 'erase-outer': 'Erase',
        exclude: 'Exclude', accept: 'Accept QC', vessel: 'Vessel',
      };
      const lbl = hintLabels[m];
      if (lbl) { hintEl.className = `m-${m}`; hintEl.textContent = lbl; }
      else      { hintEl.className = 'hidden'; }
    }

    const toasts = {
      'fiber': 'Step 1/2 — draw outer boundary (myelin)',
      'fascicle': 'Click to place points — close on 1st point or double-click',
      'exclude': 'Click to place points — excluded from N-ratio/AVF/MVF denominator',
      'paint-axon': 'Drag lasso → fills axon',
      'paint-outer': 'Drag lasso → fills myelin sheath',
      'erase-outer': 'Drag lasso → erases myelin + axon',
      'vessel': 'Drag lasso → marks blood vessel',
    };
    if (toasts[m]) this.toast(toasts[m], 'info');
    this.render();
  }

  /* ── Image list ──────────────────────────────────────────────────────── */

  async loadList() {
    if (this.gtMode) return this._loadGtList();
    const r = await fetch('/api/images');
    const imgs = await r.json();
    this.stems = imgs.map(i => i.stem);
    this.processedMap = new Map(imgs.map(i => [i.stem, i.processed]));
    const ul = $('#img-list');
    ul.innerHTML = '';
    for (const i of imgs) {
      const li = document.createElement('li');
      li.dataset.stem = i.stem;
      const dotCls = i.modified ? 'mod' : (i.processed ? 'orig' : 'raw');
      const nLabel = i.processed ? `n=${i.n_axons}` : 'raw';
      const resegBadge = i.needs_resegment ? `<span class="reseg-badge" title="Pending edits — click Recompute">⚠</span>` : '';
      li.innerHTML =
        `<span class="dot ${dotCls}"></span>` +
        `<span class="img-name" title="${i.stem}">${i.stem}</span>` +
        `${resegBadge}` +
        `<span class="img-n">${nLabel}</span>`;
      li.onclick = () => this.select(i.stem);
      ul.appendChild(li);
    }
    $('#img-count').textContent = imgs.length;
  }

  /* ── GT Mode ──────────────────────────────────────────────────────────── */

  async toggleGtMode() {
    this.gtMode = !this.gtMode;
    $('#btn-gt-toggle').classList.toggle('active', this.gtMode);
    $('#gt-banner').classList.toggle('hidden', !this.gtMode);
    // Hide production-only controls in GT mode
    $('#btn-recompute').classList.toggle('hidden', this.gtMode);
    $('#btn-reset').classList.toggle('hidden', this.gtMode);
    $('#btn-recompute-all').classList.toggle('hidden', this.gtMode);
    $$('.view-btn').forEach(b => b.classList.toggle('hidden', this.gtMode));
    $('#btn-compare').classList.toggle('hidden', this.gtMode);
    // Hide modes not relevant in GT / show GT-only modes
    $('[data-mode="accept"]').classList.toggle('hidden', this.gtMode);
    $('[data-mode="vessel"]').classList.toggle('hidden', !this.gtMode);
    // Show/hide GT validate button
    $('#gt-validate-panel').classList.toggle('hidden', !this.gtMode);

    this.cur = null;
    this.img = null;
    this.rawImg = null;
    await this.loadList();
    if (this.stems.length) {
      await this.select(this.stems[0]);
    }
  }

  async _loadGtList() {
    const r = await fetch('/api/gt/images');
    const imgs = await r.json();
    this.stems = imgs.map(i => i.stem);
    const ul = $('#img-list');
    ul.innerHTML = '';
    let nValidated = 0;
    for (const i of imgs) {
      const li = document.createElement('li');
      li.dataset.stem = i.stem;
      const validated = i.status === 'validated';
      if (validated) nValidated++;
      const dotCls = validated ? 'gt-done' : (i.n_fibers > 0 ? 'gt-wip' : 'gt-todo');
      const nLabel = i.n_fibers > 0 ? `n=${i.n_fibers}` : 'empty';
      const checkBadge = validated ? '<span class="gt-check">✓</span>' : '';
      li.innerHTML =
        `<span class="dot ${dotCls}"></span>` +
        `<span class="img-name" title="${i.stem}">${i.stem}</span>` +
        `${checkBadge}` +
        `<span class="img-n">${nLabel}</span>`;
      li.onclick = () => this.select(i.stem);
      ul.appendChild(li);
    }
    $('#img-count').textContent = imgs.length;
    $('#gt-progress').textContent = `${nValidated}/${imgs.length}`;
  }

  async select(stem) {
    if (this.gtMode) return this._gtSelect(stem);
    if (this.cur && this.cur !== stem) {
      this._updatePresenceBadge(this.cur, 0);  // clear old badge immediately
      this._leavePresence(this.cur);
    }
    this.cur = stem;
    history.replaceState(null, '', '#' + encodeURIComponent(stem));
    $('#current-stem').textContent = stem;
    this._startPresence(stem);
    $$('#img-list li').forEach(li => li.classList.toggle('active', li.dataset.stem === stem));

    this.img = null;
    this.rawImg = null;
    this.anns = [];
    this.editCount = 0;
    this.savedFasc = null;
    this.savedExcl = null;
    this._showFasciclePanel(false);
    this._showExclusionPanel(false);
    this.render();
    $('#loading').classList.remove('hidden');

    (async () => {
      const t = Date.now();
      let im = null;
      let fromRaw = false;
      try {
        im = await this._loadImg(`/api/image/${stem}/overlay?t=${t}`);
      } catch {
        try {
          im = await this._loadImg(`/api/image/${stem}/raw?t=${t}`);
          fromRaw = true;
        } catch {
          if (this.cur !== stem) return;
          $('#loading').classList.add('hidden');
          this.toast('Image not found', 'err');
          return;
        }
      }
      if (this.cur !== stem) return;
      this.img = im;
      $('#loading').classList.add('hidden');
      $('#hint').classList.add('hidden');
      if (fromRaw) {
        this.rawImg = im;
      } else {
        const rawIm = new Image();
        rawIm.onload = () => { if (this.cur === stem) { this.rawImg = rawIm; this.render(); } };
        rawIm.src = `/api/image/${stem}/raw?t=${t}`;
      }
      this.fit();
      if (fromRaw) {
        this.showMetrics({});
        this.showEditCount();
        this.render();
      } else {
        this.loadInfo(stem);
      }
      this.loadFascicle(stem);
      this.loadExclusion(stem);
    })();
  }

  _loadImg(src) {
    return new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => res(im);
      im.onerror = rej;
      im.src = src;
    });
  }

  async _gtSelect(stem) {
    if (this.cur && this.cur !== stem) {
      this._updatePresenceBadge(this.cur, 0);
      this._leavePresence(this.cur);
    }
    this.cur = stem;
    this._startPresence(stem);
    history.replaceState(null, '', '#gt:' + encodeURIComponent(stem));
    $('#current-stem').textContent = stem;
    $$('#img-list li').forEach(li => li.classList.toggle('active', li.dataset.stem === stem));

    this.img = null;
    this.rawImg = null;
    this.anns = [];
    this.editCount = 0;
    this.savedFasc = null;
    this.savedExcl = null;
    this._showFasciclePanel(false);
    this._showExclusionPanel(false);
    this.render();
    $('#loading').classList.remove('hidden');

    const t = Date.now();
    try {
      // Load raw image first
      const rawIm = await this._loadImg(`/api/gt/image/${stem}/raw?t=${t}`);
      if (this.cur !== stem) return;
      this.rawImg = rawIm;

      // Try overlay (may be empty if no annotations yet)
      try {
        const ovIm = await this._loadImg(`/api/gt/image/${stem}/overlay?t=${t}`);
        if (this.cur !== stem) return;
        this.img = ovIm;
      } catch {
        this.img = rawIm;
      }

      $('#loading').classList.add('hidden');
      $('#hint').classList.add('hidden');
      this.fit();

      // Load GT info
      const r = await fetch(`/api/gt/image/${stem}/info`);
      const info = await r.json();
      if (this.cur !== stem) return;
      this._showGtMetrics(info);
      this.loadFascicle(stem);
      this.render();
    } catch {
      if (this.cur !== stem) return;
      $('#loading').classList.add('hidden');
      this.toast('GT image not found', 'err');
    }
  }

  _showGtMetrics(info) {
    const fill = (tbl, data) => {
      const tb = tbl.querySelector('tbody');
      tb.innerHTML = '';
      for (const r of data) {
        const tr = document.createElement('tr');
        if (r[2]) tr.className = 'hl';
        tr.innerHTML = `<td>${r[0]}</td><td>${r[1]}</td>`;
        tb.appendChild(tr);
      }
    };
    const rows = [
      ['Fibers annotated', info.n_fibers, true],
      ['With axon',  info.fibers.filter(f => f.has_axon).length, true],
      ['Without axon', info.fibers.filter(f => !f.has_axon).length, false],
      ['Vessels', info.n_vessels || 0, (info.n_vessels || 0) > 0],
      ['Status', info.status === 'validated' ? '✓ Validated' : 'In progress', info.status === 'validated'],
    ];
    fill($('#tbl-metrics'), rows);
    fill($('#tbl-seg'), []);
    $('#edit-info').textContent = '';
    $('#panel').classList.remove('stale');

    // Update validate button state
    const vBtn = $('#btn-gt-validate');
    if (vBtn) {
      vBtn.textContent = info.status === 'validated' ? '✓ Validated' : 'Mark Validated';
      vBtn.classList.toggle('validated', info.status === 'validated');
    }
  }

  async gtValidate() {
    if (!this.cur || !this.gtMode) return;
    try {
      const r = await fetch(`/api/gt/image/${this.cur}/validate`, { method: 'POST' });
      const d = await r.json();
      this.toast(d.status === 'validated' ? 'Image validated' : 'Validation removed', 'ok');
      await this._loadGtList();
      // Refresh info panel
      const info = await fetch(`/api/gt/image/${this.cur}/info`).then(r => r.json());
      this._showGtMetrics(info);
    } catch (err) { this.toast(err.message, 'err'); }
  }

  async loadInfo(stem) {
    const r = await fetch(`/api/image/${stem}/info`);
    const info = await r.json();
    this.anns = [];
    this.editCount = (info.edits.deleted || []).length + (info.edits.added || []).length;
    this.showMetrics(info.metrics);
    this.showEditCount();
    this.render();
  }

  async loadFascicle(stem) {
    try {
      const prefix = this.gtMode ? '/api/gt' : '/api';
      const r = await fetch(`${prefix}/image/${stem}/fascicle?t=${Date.now()}`);
      if (!r.ok) return;
      const d = await r.json();
      if (this.cur !== stem) return;
      const list = d.contours && d.contours.length > 0 ? d.contours : null;
      this.savedFasc = list && list.length > 0 ? list : null;
      this._showFasciclePanel(!!this.savedFasc);
      this.render();
    } catch (e) { console.warn('loadFascicle:', e); }
  }

  async loadExclusion(stem) {
    try {
      const r = await fetch(`/api/image/${stem}/exclusion?t=${Date.now()}`);
      if (!r.ok) return;
      const d = await r.json();
      if (this.cur !== stem) return;
      this.savedExcl = d.contours && d.contours.length > 0 ? d.contours : null;
      this._showExclusionPanel(!!this.savedExcl);
      this.render();
    } catch (e) { console.warn('loadExclusion:', e); }
  }

  _showFasciclePanel(show) {
    $('#fascicle-panel').classList.toggle('hidden', !show);
  }

  _showExclusionPanel(show) {
    $('#exclusion-panel').classList.toggle('hidden', !show);
  }

  async clearFascicle() {
    if (!this.cur) return;
    try {
      const prefix = this.gtMode ? '/api/gt' : '/api';
      await fetch(`${prefix}/image/${this.cur}/clear-fascicle`, { method: 'POST' });
      this.savedFasc = null;
      this._showFasciclePanel(false);
      if (this.mode === 'fascicle') this.setMode('navigate');
      this.render();
      this.toast('Fascicle cleared', 'info');
    } catch (err) { this.toast(err.message, 'err'); }
  }

  async clearExclusion() {
    if (!this.cur) return;
    try {
      await fetch(`/api/image/${this.cur}/clear-exclusion`, { method: 'POST' });
      this.savedExcl = null;
      this._showExclusionPanel(false);
      if (this.mode === 'exclude') this.setMode('navigate');
      this.render();
      this.toast('Exclusion zone cleared', 'info');
    } catch (err) { this.toast(err.message, 'err'); }
  }

  /* ── Metrics panel ───────────────────────────────────────────────────── */

  showMetrics(m) {
    const fmt = (v, d = 4) => v != null ? Number(v).toFixed(d) : '--';
    const rows = [
      ['N axons',            m.n_axons ?? '--',                 true],
      ['Nerve area',         fmt(m.nerve_area_mm2) + ' mm²',    true],
      ['Excl. area',         fmt(m.exclusion_area_mm2) + ' mm²', !!(m.exclusion_area_mm2)],
      ['Total axon area',    fmt(m.total_axon_area_mm2) + ' mm²', true],
      ['Total myelin area',  fmt(m.total_myelin_area_mm2) + ' mm²', true],
      ['N-ratio',            fmt(m.nratio),                     true],
      ['G-ratio (mean)',     fmt(m.gratio_aggr),                true],
      ['G-ratio (area-w.)',  fmt(m.gratio_area_weighted),       true],
    ];
    const seg = [
      ['AVF',           fmt(m.avf)],
      ['MVF',           fmt(m.mvf)],
      ['Axon density',  fmt(m.axon_density_mm2, 0) + ' /mm²'],
    ];
    const fill = (tbl, data) => {
      const tb = tbl.querySelector('tbody');
      tb.innerHTML = '';
      for (const r of data) {
        const tr = document.createElement('tr');
        if (r[2]) tr.className = 'hl';
        tr.innerHTML = `<td>${r[0]}</td><td>${r[1]}</td>`;
        tb.appendChild(tr);
      }
    };
    fill($('#tbl-metrics'), rows);
    fill($('#tbl-seg'), seg);
  }

  showEditCount() {
    const stale = this.editCount > 0;
    $('#edit-info').textContent = stale
      ? `${this.editCount} edit(s) -- recompute for metrics`
      : '';
    $('#panel').classList.toggle('stale', stale);
  }

  /* ── Canvas ──────────────────────────────────────────────────────────── */

  fit() {
    if (!this.img) return;
    const cw = this.cvs.width, ch = this.cvs.height;
    const iw = this.img.naturalWidth, ih = this.img.naturalHeight;
    this.zoom = Math.min(cw / iw, ch / ih) * 0.95;
    this.panX = (cw - iw * this.zoom) / 2;
    this.panY = (ch - ih * this.zoom) / 2;
    this.render();
  }

  render() {
    const ctx = this.ctx;
    const cw = this.cvs.width, ch = this.cvs.height;
    ctx.clearRect(0, 0, cw, ch);
    ctx.save();
    ctx.translate(this.panX, this.panY);
    ctx.scale(this.zoom, this.zoom);

    // Draw raw then overlay with opacity
    if (this.rawImg && this.img && this.img !== this.rawImg) {
      ctx.drawImage(this.rawImg, 0, 0);
      ctx.globalAlpha = this.overlayOpacity;
      ctx.drawImage(this.img, 0, 0);
      ctx.globalAlpha = 1;
    } else if (this.img) {
      ctx.drawImage(this.img, 0, 0);
    }

    // ── Saved exclusion zone (orange hatch) ───────────────────────────────
    if (this.savedExcl && this.savedExcl.length > 0 && this.exclPts.length === 0) {
      ctx.save();
      for (const contour of this.savedExcl) {
        if (contour.length < 3) continue;
        ctx.beginPath();
        ctx.moveTo(contour[0][0], contour[0][1]);
        for (let i = 1; i < contour.length; i++) ctx.lineTo(contour[i][0], contour[i][1]);
        ctx.closePath();
        ctx.fillStyle = 'rgba(230,126,34,0.18)';
        ctx.fill();
        ctx.strokeStyle = '#E67E22';
        ctx.lineWidth = 1.5 / this.zoom;
        ctx.setLineDash([8 / this.zoom, 5 / this.zoom]);
        ctx.shadowBlur = 4; ctx.shadowColor = 'rgba(0,0,0,0.7)';
        ctx.stroke();
      }
      ctx.setLineDash([]); ctx.shadowBlur = 0;
      const top = this.savedExcl[0].reduce((a, b) => a[1] < b[1] ? a : b);
      ctx.font = `bold ${Math.max(10, 12 / this.zoom)}px sans-serif`;
      ctx.fillStyle = '#E67E22'; ctx.textAlign = 'center';
      ctx.fillText('EXCLU', top[0], top[1] - 6 / this.zoom);
      ctx.restore();
    }

    // ── Saved fascicle boundary (yellow dashed) ───────────────────────────
    if (this.savedFasc && this.savedFasc.length > 0 && this.fascPts.length === 0) {
      ctx.save();
      for (const contour of this.savedFasc) {
        if (contour.length < 3) continue;
        ctx.beginPath();
        ctx.moveTo(contour[0][0], contour[0][1]);
        for (let i = 1; i < contour.length; i++) ctx.lineTo(contour[i][0], contour[i][1]);
        ctx.closePath();
        ctx.fillStyle = 'rgba(241,196,15,0.07)';
        ctx.fill();
        ctx.strokeStyle = '#F1C40F';
        ctx.lineWidth = 1.5 / this.zoom;
        ctx.setLineDash([10 / this.zoom, 6 / this.zoom]);
        ctx.shadowBlur = 6; ctx.shadowColor = 'rgba(0,0,0,0.8)';
        ctx.stroke();
      }
      ctx.setLineDash([]); ctx.shadowBlur = 3;
      const top = this.savedFasc[0].reduce((a, b) => a[1] < b[1] ? a : b);
      const label = this.savedFasc.length > 1 ? `FASCICLE ×${this.savedFasc.length}` : 'FASCICLE';
      ctx.font = `bold ${Math.max(11, 13 / this.zoom)}px sans-serif`;
      ctx.fillStyle = '#F1C40F'; ctx.textAlign = 'center';
      ctx.fillText(label, top[0], top[1] - 8 / this.zoom);
      ctx.restore();
    }

    // ── Annotation markers ────────────────────────────────────────────────
    const r = 14 / this.zoom, lw = 1.5 / this.zoom;
    for (const a of this.anns) {
      ctx.save();
      // Pending — full-size dashed marker (amber) so user knows what's queued
      if (a.t === 'del_pending') {
        ctx.strokeStyle = 'rgba(255,167,38,0.9)'; ctx.lineWidth = lw;
        ctx.setLineDash([4 / this.zoom, 3 / this.zoom]);
        ctx.beginPath(); ctx.arc(a.x, a.y, r, 0, Math.PI * 2); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(a.x - r * .55, a.y - r * .55); ctx.lineTo(a.x + r * .55, a.y + r * .55);
        ctx.moveTo(a.x + r * .55, a.y - r * .55); ctx.lineTo(a.x - r * .55, a.y + r * .55);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore(); continue;
      }
      if (a.t === 'add_pending') {
        ctx.strokeStyle = 'rgba(255,167,38,0.9)'; ctx.lineWidth = lw;
        ctx.setLineDash([4 / this.zoom, 3 / this.zoom]);
        ctx.beginPath(); ctx.arc(a.x, a.y, r, 0, Math.PI * 2); ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore(); continue;
      }
      if (a.t === 'del') {
        ctx.strokeStyle = '#ef5350'; ctx.lineWidth = lw;
        ctx.beginPath(); ctx.arc(a.x, a.y, r, 0, Math.PI * 2); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(a.x - r * .55, a.y - r * .55); ctx.lineTo(a.x + r * .55, a.y + r * .55);
        ctx.moveTo(a.x + r * .55, a.y - r * .55); ctx.lineTo(a.x - r * .55, a.y + r * .55);
        ctx.stroke();
      } else if (a.t === 'fiber') {
        ctx.lineWidth = lw;
        ctx.strokeStyle = '#9b59b6';
        ctx.beginPath(); ctx.arc(a.x, a.y, r * 1.3, 0, Math.PI * 2); ctx.stroke();
        ctx.strokeStyle = '#66bb6a';
        ctx.beginPath(); ctx.arc(a.x, a.y, r * .65, 0, Math.PI * 2); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(a.x, a.y - r * .45); ctx.lineTo(a.x, a.y + r * .45);
        ctx.moveTo(a.x - r * .45, a.y); ctx.lineTo(a.x + r * .45, a.y);
        ctx.stroke();
      } else {
        ctx.strokeStyle = '#66bb6a'; ctx.lineWidth = lw;
        ctx.beginPath(); ctx.arc(a.x, a.y, r, 0, Math.PI * 2); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(a.x, a.y - r * .55); ctx.lineTo(a.x, a.y + r * .55);
        ctx.moveTo(a.x - r * .55, a.y); ctx.lineTo(a.x + r * .55, a.y);
        ctx.stroke();
      }
      ctx.restore();
    }

    // ── Stored outer polygon (fiber mode step 2) — violet dashed ─────────
    if (this.outerPoly.length > 1) {
      ctx.save();
      ctx.strokeStyle = '#9b59b6'; ctx.lineWidth = 1 / this.zoom;
      ctx.setLineDash([6 / this.zoom, 4 / this.zoom]);
      ctx.beginPath();
      ctx.moveTo(this.outerPoly[0].x, this.outerPoly[0].y);
      for (let i = 1; i < this.outerPoly.length; i++)
        ctx.lineTo(this.outerPoly[i].x, this.outerPoly[i].y);
      ctx.closePath();
      ctx.fillStyle = 'rgba(155,89,182,0.10)'; ctx.fill(); ctx.stroke();
      ctx.restore();
    }

    // ── Active drawing lasso (thin 1px line) ──────────────────────────────
    if (this.drawPts.length > 1) {
      const clr = this._lassoColor();
      const bg  = this._lassoBg();
      ctx.save();
      ctx.strokeStyle = clr; ctx.lineWidth = 1 / this.zoom;
      ctx.setLineDash([5 / this.zoom, 3 / this.zoom]);
      ctx.beginPath();
      ctx.moveTo(this.drawPts[0].x, this.drawPts[0].y);
      for (let i = 1; i < this.drawPts.length; i++)
        ctx.lineTo(this.drawPts[i].x, this.drawPts[i].y);
      ctx.closePath();
      ctx.fillStyle = bg; ctx.fill(); ctx.stroke();
      ctx.restore();
    }

    // ── Fascicle polygon (click-to-add vertices, yellow) ─────────────────
    this._drawClickPolygon(this.fascPts, this.fascMouse, '#F1C40F', 'rgba(241,196,15,0.10)');

    // ── Exclusion polygon (click-to-add vertices, orange) ────────────────
    this._drawClickPolygon(this.exclPts, this.exclMouse, '#E67E22', 'rgba(230,126,34,0.12)');

    ctx.restore();

    // ── Magnifying loupe — fascicle/exclude mode ──────────────────────────
    const activeMouse = this.mode === 'fascicle' ? this.fascMouse
                      : this.mode === 'exclude'   ? this.exclMouse : null;
    const loupeImg = this.rawImg || this.img;
    if (activeMouse && loupeImg) {
      const mx = activeMouse.x, my = activeMouse.y;
      const sX = mx * this.zoom + this.panX;
      const sY = my * this.zoom + this.panY;
      const LR = 72, SRC = 52;
      let lX = sX + LR + 14, lY = sY - LR - 14;
      if (lX + LR > cw) lX = sX - LR - 14;
      if (lY - LR < 0)  lY = sY + LR + 14;

      ctx.save();
      ctx.shadowBlur = 14; ctx.shadowColor = 'rgba(0,0,0,0.65)';
      ctx.beginPath(); ctx.arc(lX, lY, LR + 1, 0, Math.PI * 2);
      ctx.fillStyle = '#000'; ctx.fill(); ctx.shadowBlur = 0;
      ctx.beginPath(); ctx.arc(lX, lY, LR, 0, Math.PI * 2); ctx.clip();
      const sx = Math.max(0, mx - SRC);
      const sy = Math.max(0, my - SRC);
      const sw = Math.min(loupeImg.naturalWidth  - sx, SRC * 2);
      const sh = Math.min(loupeImg.naturalHeight - sy, SRC * 2);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(loupeImg, sx, sy, sw, sh, lX - LR, lY - LR, LR * 2, LR * 2);
      ctx.imageSmoothingEnabled = true;
      const ringClr = this.mode === 'exclude' ? '#E67E22' : '#F1C40F';
      ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(lX - 10, lY); ctx.lineTo(lX + 10, lY);
      ctx.moveTo(lX, lY - 10); ctx.lineTo(lX, lY + 10);
      ctx.stroke();
      ctx.restore();
      ctx.beginPath(); ctx.arc(lX, lY, LR, 0, Math.PI * 2);
      ctx.strokeStyle = ringClr; ctx.lineWidth = 2; ctx.stroke();
    }

    $('#zoom-info').textContent = `${(this.zoom * 100).toFixed(0)}%`;
  }

  /* Draw a click-polygon (shared by fascicle and exclude modes). */
  _drawClickPolygon(pts, mouse, stroke, fill) {
    if (pts.length === 0) return;
    const ctx = this.ctx;
    const lw = 1 / this.zoom;
    ctx.save();
    if (pts.length >= 3) {
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
      if (mouse) ctx.lineTo(mouse.x, mouse.y);
      ctx.closePath();
      ctx.fillStyle = fill; ctx.fill();
    }
    if (pts.length > 1) {
      ctx.setLineDash([]);
      ctx.strokeStyle = stroke; ctx.lineWidth = lw;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.stroke();
    }
    if (mouse) {
      const last = pts[pts.length - 1];
      ctx.setLineDash([4 / this.zoom, 3 / this.zoom]);
      ctx.strokeStyle = stroke + 'b0'; ctx.lineWidth = lw;
      ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(mouse.x, mouse.y); ctx.stroke();
    }
    ctx.setLineDash([]);
    for (let i = 0; i < pts.length; i++) {
      const pr = (i === 0 ? 5 : 3) / this.zoom;
      ctx.beginPath(); ctx.arc(pts[i].x, pts[i].y, pr, 0, Math.PI * 2);
      ctx.fillStyle = i === 0 ? '#E67E22' : stroke; ctx.fill();
    }
    if (pts.length >= 3 && mouse) {
      const snapR = 14 / this.zoom;
      if (Math.hypot(mouse.x - pts[0].x, mouse.y - pts[0].y) < snapR) {
        ctx.beginPath(); ctx.arc(pts[0].x, pts[0].y, snapR, 0, Math.PI * 2);
        ctx.strokeStyle = stroke + '99'; ctx.lineWidth = lw;
        ctx.setLineDash([3 / this.zoom, 3 / this.zoom]); ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    ctx.restore();
  }

  _lassoColor() {
    if (this.mode === 'paint-axon') return '#66bb6a';
    if (this.mode === 'paint-outer') return '#9b59b6';
    if (this.mode === 'erase-outer') return '#ef5350';
    if (this.mode === 'vessel') return '#e74c3c';
    if (this.mode === 'fiber' && this.fiberStep === 0) return '#9b59b6';
    if (this.mode === 'fiber' && this.fiberStep === 1) return '#66bb6a';
    return '#4fc3f7';
  }

  _lassoBg() {
    if (this.mode === 'paint-axon') return 'rgba(102,187,106,0.15)';
    if (this.mode === 'paint-outer') return 'rgba(155,89,182,0.12)';
    if (this.mode === 'erase-outer') return 'rgba(239,83,80,0.12)';
    if (this.mode === 'vessel') return 'rgba(231,76,60,0.15)';
    if (this.mode === 'fiber' && this.fiberStep === 0) return 'rgba(155,89,182,0.12)';
    if (this.mode === 'fiber' && this.fiberStep === 1) return 'rgba(102,187,106,0.15)';
    return 'rgba(79,195,247,0.12)';
  }

  s2i(e) {
    const rc = this.cvs.getBoundingClientRect();
    return {
      x: Math.round((e.clientX - rc.left - this.panX) / this.zoom),
      y: Math.round((e.clientY - rc.top  - this.panY) / this.zoom),
    };
  }

  /* ── Pointer (mouse + touch + pen) ────────────────────────────────────── */

  onDown(e) {
    e.preventDefault();
    this._pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, type: e.pointerType });

    // 2+ pointers → pinch-zoom (cancel any in-progress draw)
    if (this._pointers.size >= 2) {
      this.drawing = false; this.drawPts = [];
      this.dragging = false; $('#viewer').classList.remove('panning');
      this._initPinch();
      return;
    }

    // Touch finger → always pan (pencil/mouse use the active tool)
    if (e.pointerType === 'touch') {
      this.dragging = true;
      this.lastM = { x: e.clientX, y: e.clientY };
      $('#viewer').classList.add('panning');
      return;
    }

    // Mouse/Pen: right-click, middle-click, space, or navigate mode → pan
    const isR = e.button === 2, isM = e.button === 1;
    if (isR || isM || this.spaceHeld || this.mode === 'navigate') {
      this.dragging = true;
      this.lastM = { x: e.clientX, y: e.clientY };
      $('#viewer').classList.add('panning');
      return;
    }

    // Tool actions (mouse left-click or pen tap)
    if (this.mode === 'delete') {
      // On touch/tablet devices (coarse pointer), require double-tap to confirm
      const isCoarse = matchMedia('(pointer: coarse)').matches;
      if (isCoarse) {
        const pt = this.s2i(e);
        const now = Date.now();
        if (this._delConfirm && now - this._delConfirm.t < 1500 &&
            Math.hypot(pt.x - this._delConfirm.x, pt.y - this._delConfirm.y) < 30) {
          this._delConfirm = null;
          this.clickDelete(e);
        } else {
          this._delConfirm = { x: pt.x, y: pt.y, t: now };
          this.toast('Tap again to confirm delete', 'info');
        }
      } else {
        this.clickDelete(e);
      }
    }

    if (this.mode === 'accept') { this.clickAccept(e); return; }

    // Freehand lasso modes
    if (['fiber', 'paint-axon', 'paint-outer', 'erase-outer', 'vessel'].includes(this.mode)) {
      this.drawing = true;
      this.drawPts = [this.s2i(e)];
    }

    // Click-polygon modes
    if (this.mode === 'fascicle') this._polygonClick(e, 'fasc');
    if (this.mode === 'exclude')  this._polygonClick(e, 'excl');
  }

  _polygonClick(e, key) {
    const pts  = key === 'fasc' ? this.fascPts  : this.exclPts;
    const setP = key === 'fasc'
      ? p => { this.fascPts = p; }
      : p => { this.exclPts = p; };
    const lastT = key === 'fasc' ? '_lastFascClick' : '_lastExclClick';

    const pt = this.s2i(e);
    const now = Date.now();
    const dbl = now - this[lastT] < 350;
    this[lastT] = now;

    const submit = key === 'fasc'
      ? () => this.submitFascicle([...pts])
      : () => this.submitExclusion([...pts]);

    if (dbl && pts.length >= 3) {
      submit(); setP([]);
      if (key === 'fasc') this.fascMouse = null;
      else this.exclMouse = null;
    } else if (pts.length >= 3 &&
        Math.hypot(pt.x - pts[0].x, pt.y - pts[0].y) < 14 / this.zoom) {
      submit(); setP([]);
      if (key === 'fasc') this.fascMouse = null;
      else this.exclMouse = null;
    } else {
      pts.push(pt);
    }
    this.render();
  }

  onMove(e) {
    // Update tracked pointer
    if (this._pointers.has(e.pointerId)) {
      this._pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, type: e.pointerType });
    }

    // Pinch-zoom in progress
    if (this._pointers.size >= 2 && this._pinchDist > 0) {
      this._handlePinch(); return;
    }

    const pt = this.s2i(e);
    $('#coords').textContent = `x=${pt.x}  y=${pt.y}`;
    this._moveCursorHint(e.clientX, e.clientY);

    if (this.dragging) {
      this.panX += e.clientX - this.lastM.x;
      this.panY += e.clientY - this.lastM.y;
      this.lastM = { x: e.clientX, y: e.clientY };
      this.render(); return;
    }

    // Polygon preview — works with pen hover and mouse, not touch (no hover)
    if (this.mode === 'fascicle') {
      this.fascMouse = pt;
      if (this.fascPts.length > 0) this.render();
      return;
    }
    if (this.mode === 'exclude') {
      this.exclMouse = pt;
      if (this.exclPts.length > 0) this.render();
      return;
    }
    if (this.drawing) { this.drawPts.push(pt); this.render(); }
  }

  onUp(e) {
    this._pointers.delete(e.pointerId);

    // Exiting pinch → reset, optionally switch to single-finger pan
    if (this._pinchDist > 0 && this._pointers.size < 2) {
      this._pinchDist = 0; this._pinchMid = null;
      if (this._pointers.size === 1) {
        const p = [...this._pointers.values()][0];
        this.dragging = true;
        this.lastM = { x: p.x, y: p.y };
      }
      return;
    }

    if (this.dragging) {
      this.dragging = false;
      $('#viewer').classList.remove('panning');
      return;
    }
    if (this.drawing && this.drawPts.length >= 3) {
      // Sample to reduce point count
      const sampled = this.drawPts.filter((_, i) => i % 3 === 0 || i === this.drawPts.length - 1);
      if (this.mode === 'fiber') {
        if (this.fiberStep === 0) {
          this.outerPoly = sampled;
          this.fiberStep = 1;
          this.drawing = false; this.drawPts = [];
          this.render();
          this.toast('Step 2/2 : draw the axon inside', 'info');
          return;
        } else {
          this.submitFiber(sampled);
        }
      } else if (this.mode === 'paint-axon') {
        this.submitPaintAxon(sampled);
      } else if (this.mode === 'paint-outer') {
        this.submitPaintOuter(sampled);
      } else if (this.mode === 'erase-outer') {
        this.submitEraseOuter(sampled);
      } else if (this.mode === 'vessel') {
        this.submitVessel(sampled);
      }
    }
    this.drawing = false; this.drawPts = [];
    this.render();
  }

  onWheel(e) {
    e.preventDefault();
    const rc = this.cvs.getBoundingClientRect();
    const mx = e.clientX - rc.left, my = e.clientY - rc.top;
    const f = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const nz = Math.max(0.02, Math.min(80, this.zoom * f));
    this.panX = mx - (mx - this.panX) * (nz / this.zoom);
    this.panY = my - (my - this.panY) * (nz / this.zoom);
    this.zoom = nz;
    this.render();
  }

  /* ── Pinch-to-zoom (two-finger touch) ──────────────────────────────── */

  _initPinch() {
    const pts = [...this._pointers.values()];
    this._pinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y) || 1;
    this._pinchMid = {
      x: (pts[0].x + pts[1].x) / 2,
      y: (pts[0].y + pts[1].y) / 2,
    };
  }

  _handlePinch() {
    const pts = [...this._pointers.values()];
    if (pts.length < 2) return;

    const newDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y) || 1;
    const newMid = {
      x: (pts[0].x + pts[1].x) / 2,
      y: (pts[0].y + pts[1].y) / 2,
    };

    // Zoom centered on pinch midpoint
    const scale = newDist / this._pinchDist;
    const rc = this.cvs.getBoundingClientRect();
    const mx = this._pinchMid.x - rc.left;
    const my = this._pinchMid.y - rc.top;
    const nz = Math.max(0.02, Math.min(80, this.zoom * scale));
    this.panX = mx - (mx - this.panX) * (nz / this.zoom);
    this.panY = my - (my - this.panY) * (nz / this.zoom);
    this.zoom = nz;

    // Pan with midpoint drift
    this.panX += newMid.x - this._pinchMid.x;
    this.panY += newMid.y - this._pinchMid.y;

    this._pinchDist = newDist;
    this._pinchMid = newMid;
    this.render();
  }

  onKey(e, down) {
    if (e.code === 'Space') {
      this.spaceHeld = down;
      if (down) {
        e.preventDefault();
        if (this.drawing) { this.drawing = false; this.drawPts = []; this.render(); }
        $('#viewer').classList.add('space-held');
        const h = document.getElementById('cursor-hint'); if (h) h.style.display = 'none';
      } else {
        $('#viewer').classList.remove('space-held');
        const h = document.getElementById('cursor-hint');
        if (h && !h.classList.contains('hidden')) h.style.display = '';
      }
      return;
    }
    if (!down) return;
    if ((e.ctrlKey || e.metaKey) && e.key === 'z') { e.preventDefault(); this.undo(); return; }
    if (e.key.length === 1) {
      this._keyBuffer = (this._keyBuffer + e.key.toLowerCase()).slice(-5);
      if (this._keyBuffer === 'marie') { this._keyBuffer = ''; this._marie('💙 Coucou Marie — bonne analyse !'); }
    }
    if (e.key === '1') this.setMode('navigate');
    if (e.key === '2') this.setMode('delete');
    if (e.key === '3') this.setMode('paint-axon');
    if (e.key === '4') this.setMode('paint-outer');
    if (e.key === '5') this.setMode('erase-outer');
    if (e.key === '6') this.setMode('fiber');
    if (e.key === '7') this.setMode('fascicle');
    if (e.key === '8') this.setMode('exclude');
    if (e.key === '9') this.setMode('accept');
    if (e.key === '0' && this.gtMode) this.setMode('vessel');
    if (e.key === 'Escape') {
      if (this.mode === 'fiber' && (this.drawing || this.fiberStep > 0)) {
        this.fiberStep = 0; this.outerPoly = []; this.drawPts = [];
        this.drawing = false; this.render(); this.toast('Cancelled', 'info');
      }
      if (['paint-axon', 'paint-outer', 'erase-outer', 'vessel'].includes(this.mode) && this.drawing) {
        this.drawing = false; this.drawPts = []; this.render(); this.toast('Cancelled', 'info');
      }
      if (this.mode === 'fascicle' && this.fascPts.length > 0) {
        this.fascPts = []; this.fascMouse = null; this.render(); this.toast('Cancelled', 'info');
      }
      if (this.mode === 'exclude' && this.exclPts.length > 0) {
        this.exclPts = []; this.exclMouse = null; this.render(); this.toast('Cancelled', 'info');
      }
    }
    if (e.key === 'f' || e.key === 'F') this.fit();
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { this.nav(1); e.preventDefault(); }
    if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')   { this.nav(-1); e.preventDefault(); }
  }

  nav(d) {
    if (!this.cur || !this.stems.length) return;
    const i = this.stems.indexOf(this.cur);
    const n = Math.max(0, Math.min(this.stems.length - 1, i + d));
    if (n !== i) this.select(this.stems[n]);
  }

  /* ── Actions ─────────────────────────────────────────────────────────── */

  showBusy(on, msg) {
    const el = $('#loading');
    if (on) { el.textContent = msg || 'Processing...'; el.classList.remove('hidden'); }
    else     { el.classList.add('hidden'); }
  }

  reloadOverlay() {
    const stem = this.cur;
    if (!stem) return;
    const prefix = this.gtMode ? '/api/gt' : '/api';
    const im = new Image();
    im.onload = () => { if (this.cur !== stem) return; this.img = im; this.anns = []; this.showBusy(false); this.render(); };
    im.onerror = () => this.showBusy(false);
    im.src = `${prefix}/image/${stem}/overlay?t=${Date.now()}`;
  }

  async clickDelete(e) {
    if (!this.cur) return;
    const stem = this.cur;
    const pt = this.s2i(e);
    const pAnn = { t: 'del_pending', x: pt.x, y: pt.y };
    this.anns.push(pAnn); this.render();
    this._enqueue(async () => {
      if (this.cur !== stem) { this._removeAnn(pAnn); return; }
      let resp;
      const prefix = this.gtMode ? '/api/gt' : '/api';
      try {
        resp = await fetch(`${prefix}/image/${stem}/delete?refresh=false`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ x: pt.x, y: pt.y }),
        });
      } catch (err) { this._removeAnn(pAnn); this.toast(err.message, 'err'); return; }
      this._removeAnn(pAnn);
      if (!resp.ok) {
        this.toast((await resp.json()).detail || 'Nothing here', 'err');
        this.render(); return;
      }
      const d = await resp.json();
      this.anns.push({ t: 'del', x: d.x, y: d.y, label: d.deleted });
      this.editCount++; this.showEditCount();
      this._qDirty = true; this.render();
    }, 'Delete');
  }

  async clickAccept(e) {
    if (!this.cur) return;
    const pt = this.s2i(e);
    try {
      const r = await fetch(`/api/image/${this.cur}/qc-accept`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x: pt.x, y: pt.y }),
      });
      if (!r.ok) { this.toast((await r.json()).detail || 'No fiber here', 'err'); return; }
      const d = await r.json();
      const msg = d.status === 'added'
        ? `Fiber #${d.label} accepted — Recompute to apply`
        : `Fiber #${d.label} un-accepted`;
      this.toast(msg, 'info');
      // Mark panel stale so "Recompute to refresh" banner shows
      document.getElementById('panel')?.classList.add('stale');
    } catch (err) { this.toast(err.message, 'err'); }
  }

  async submitFiber(innerPts) {
    if (!this.cur) return;
    const outer = this.outerPoly;
    if (outer.length < 3 || innerPts.length < 3) { this.toast('Draw larger shapes', 'err'); return; }
    this.fiberStep = 0; this.outerPoly = [];
    this.showBusy(true, 'Adding fiber...');
    this.render();
    try {
      const prefix = this.gtMode ? '/api/gt' : '/api';
      const r = await fetch(`${prefix}/image/${this.cur}/add-fiber`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ outer_points: outer.map(p => [p.x, p.y]), inner_points: innerPts.map(p => [p.x, p.y]) }),
      });
      if (!r.ok) { this.showBusy(false); this.toast((await r.json()).detail || 'Error', 'err'); return; }
      const d = await r.json();
      this.anns.push({ t: 'fiber', x: d.x, y: d.y, label: d.added });
      this.editCount++; this.showEditCount();
      this.toast(`Fiber #${d.added} added`, 'ok');
      if (d.refreshed) this.reloadOverlay(); else this.showBusy(false);
      if (this.gtMode) {
        this._loadGtList();
        fetch(`/api/gt/image/${this.cur}/info`).then(r => r.json()).then(info => this._showGtMetrics(info)).catch(() => {});
      }
    } catch (err) { this.showBusy(false); this.toast(err.message, 'err'); }
  }

  async submitPaintAxon(pts) {
    if (!this.cur) return;
    const stem = this.cur;
    const pAnn = { t: 'add_pending', ...this._ptCentroid(pts) };
    this.anns.push(pAnn); this.render();
    this._enqueue(async () => {
      if (this.cur !== stem) { this._removeAnn(pAnn); return; }
      let resp;
      const prefix = this.gtMode ? '/api/gt' : '/api';
      const endpoint = this.gtMode ? 'paint-axon' : 'add';
      try {
        resp = await fetch(`${prefix}/image/${stem}/${endpoint}?refresh=false`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ points: pts.map(p => [p.x, p.y]) }),
        });
      } catch (err) { this._removeAnn(pAnn); this.toast(err.message, 'err'); return; }
      this._removeAnn(pAnn);
      if (!resp.ok) { this.toast((await resp.json()).detail || 'Error', 'err'); this.render(); return; }
      const d = await resp.json();
      this.anns.push({ t: 'add', x: d.x, y: d.y });
      this.editCount++; this.showEditCount();
      this._qDirty = true; this.render();
    }, '+Axon');
  }

  async submitPaintOuter(pts) {
    if (!this.cur) return;
    const stem = this.cur;
    const pAnn = { t: 'add_pending', ...this._ptCentroid(pts) };
    this.anns.push(pAnn); this.render();
    this._enqueue(async () => {
      if (this.cur !== stem) { this._removeAnn(pAnn); return; }
      let resp;
      const prefix = this.gtMode ? '/api/gt' : '/api';
      try {
        resp = await fetch(`${prefix}/image/${stem}/paint-outer?refresh=false`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ points: pts.map(p => [p.x, p.y]) }),
        });
      } catch (err) { this._removeAnn(pAnn); this.toast(err.message, 'err'); return; }
      this._removeAnn(pAnn);
      if (!resp.ok) { this.toast((await resp.json()).detail || 'Error', 'err'); this.render(); return; }
      const d = await resp.json();
      this.anns.push({ t: 'fiber', x: d.x, y: d.y });
      this.editCount++; this.showEditCount();
      this._qDirty = true; this.render();
    }, '+Myelin');
  }

  async submitEraseOuter(pts) {
    if (!this.cur) return;
    const stem = this.cur;
    const pAnn = { t: 'del_pending', ...this._ptCentroid(pts) };
    this.anns.push(pAnn); this.render();
    this._enqueue(async () => {
      if (this.cur !== stem) { this._removeAnn(pAnn); return; }
      let resp;
      const prefix = this.gtMode ? '/api/gt' : '/api';
      const eraseEndpoint = this.gtMode ? 'erase' : 'erase-outer';
      try {
        resp = await fetch(`${prefix}/image/${stem}/${eraseEndpoint}?refresh=false`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ points: pts.map(p => [p.x, p.y]) }),
        });
      } catch (err) { this._removeAnn(pAnn); this.toast(err.message, 'err'); return; }
      this._removeAnn(pAnn);
      if (!resp.ok) { this.toast((await resp.json()).detail || 'Error', 'err'); this.render(); return; }
      this.editCount++; this.showEditCount();
      this._qDirty = true; this.render();
    }, 'Erase');
  }

  async submitVessel(pts) {
    if (!this.cur) return;
    const stem = this.cur;
    const pAnn = { t: 'add_pending', ...this._ptCentroid(pts) };
    this.anns.push(pAnn); this.render();
    this._enqueue(async () => {
      if (this.cur !== stem) { this._removeAnn(pAnn); return; }
      let resp;
      try {
        resp = await fetch(`/api/gt/image/${stem}/add-vessel?refresh=false`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ points: pts.map(p => [p.x, p.y]) }),
        });
      } catch (err) { this._removeAnn(pAnn); this.toast(err.message, 'err'); return; }
      this._removeAnn(pAnn);
      if (!resp.ok) { this.toast((await resp.json()).detail || 'Error', 'err'); this.render(); return; }
      const d = await resp.json();
      this.anns.push({ t: 'add', x: d.x, y: d.y, label: d.added });
      this.editCount++; this.showEditCount();
      this._qDirty = true; this.render();
    }, 'Vessel');
  }

  async submitFascicle(pts) {
    if (!this.cur) return;
    if (pts.length < 3) { this.toast('Draw a larger shape', 'err'); return; }
    this.showBusy(true, 'Saving fascicle...');
    try {
      const prefix = this.gtMode ? '/api/gt' : '/api';
      const r = await fetch(`${prefix}/image/${this.cur}/set-fascicle`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ points: pts.map(p => [p.x, p.y]) }),
      });
      if (!r.ok) { this.showBusy(false); this.toast((await r.json()).detail || 'Error', 'err'); return; }
      this.showBusy(false);
      this.toast(this.gtMode ? 'Fascicle saved' : 'Fascicle saved — click Recompute to apply', 'ok');
      this.loadFascicle(this.cur);
      this.setMode('navigate');
    } catch (err) { this.showBusy(false); this.toast(err.message, 'err'); }
  }

  async submitExclusion(pts) {
    if (!this.cur) return;
    if (pts.length < 3) { this.toast('Draw a larger shape', 'err'); return; }
    this.showBusy(true, 'Saving exclusion...');
    try {
      const r = await fetch(`/api/image/${this.cur}/set-exclusion`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ points: pts.map(p => [p.x, p.y]) }),
      });
      if (!r.ok) { this.showBusy(false); this.toast((await r.json()).detail || 'Error', 'err'); return; }
      this.showBusy(false);
      this.toast('Exclusion zone saved — click Recompute to apply', 'ok');
      this.loadExclusion(this.cur);
      this.setMode('navigate');
    } catch (err) { this.showBusy(false); this.toast(err.message, 'err'); }
  }

  async recompute() {
    if (!this.cur) return;
    this.toast('Recomputing...', 'info');
    $('#btn-recompute').disabled = true;
    try {
      const r = await fetch(`/api/image/${this.cur}/recompute`, { method: 'POST' });
      if (!r.ok) { this.toast((await r.json()).detail || 'Error', 'err'); return; }
      const d = await r.json();
      this.anns = []; this.editCount = 0;
      await this.select(this.cur);
      await this.loadList();
      // Toast AFTER UI has settled so it's visible on the fresh canvas
      const agg = d.agg || {};
      const fmt = (v, n = 3) => v != null ? Number(v).toFixed(n) : '--';
      this.toast(
        `✓ Done — ${d.n_axons} axons  |  g=${fmt(agg.gratio_aggr)}  |  N=${fmt(agg.nratio)}`,
        'ok', 6000
      );
      this._flashPanel();
    } catch (err) { this.toast(err.message, 'err'); }
    finally { $('#btn-recompute').disabled = false; }
  }

  async reset() {
    if (!this.cur) return;
    if (!confirm('Reset to original segmentation? All pending edits will be lost.')) return;
    try {
      const r = await fetch(`/api/image/${this.cur}/reset`, { method: 'POST' });
      const d = await r.json();
      if (d.status === 'no_backup') { this.toast('Nothing to reset', 'info'); return; }
      this.toast('Reset OK', 'ok');
      this.anns = []; this.editCount = 0;
      await this.select(this.cur);
      await this.loadList();
    } catch (err) { this.toast(err.message, 'err'); }
  }

  async undo() {
    if (!this.cur) return;
    // Flush pending queue items so they don't overwrite the restored state
    if (this._qItems.length > 0) {
      this._qItems = [];
      this._qDirty = false;
      this._qLog = [];
      this._renderStack();
      this._qBarDone();
    }
    const prefix = this.gtMode ? '/api/gt' : '/api';
    try {
      const r = await fetch(`${prefix}/image/${this.cur}/undo`, { method: 'POST' });
      if (!r.ok) { this.toast((await r.json()).detail || 'Nothing to undo', 'info'); return; }
      this.toast('Undo OK', 'ok');
      if (this.editCount > 0) this.editCount--;
      this.anns.pop();
      this.showEditCount();
      this.reloadOverlay();
      if (this.gtMode) this._loadGtList();  // refresh counts
    } catch (err) { this.toast(err.message, 'err'); }
  }

  async recomputeAll() {
    const nProcessed = [...this.processedMap.values()].filter(Boolean).length;
    const secsEst = Math.max(10, nProcessed * 8);
    const timeStr = secsEst >= 60
      ? `~${Math.round(secsEst / 60)} min`
      : `~${secsEst} sec`;
    const msg =
      `Recompute morphometrics for ${nProcessed} image(s)?\n\n` +
      `⏱  Estimated: ${timeStr} (QC + visualisations only, no Cellpose)\n\n` +
      `✅  The app stays fully usable during the computation.`;
    if (!confirm(msg)) return;
    const btn = $('#btn-recompute-all');
    btn.disabled = true;
    try {
      const r = await fetch('/api/recompute-all', { method: 'POST' });
      const d = await r.json();
      if (d.status === 'already_running') { this.toast('Batch recompute already running', 'info'); }
      else if (d.status === 'started') { this.toast(`Recomputing ${d.total} images in background...`, 'info'); }
      else { this.toast('No images found', 'info'); btn.disabled = false; return; }
      this._pollBatch();
    } catch (err) { this.toast(err.message, 'err'); btn.disabled = false; }
  }

  _pollBatch() {
    const prog = $('#batch-progress');
    const btn = $('#btn-recompute-all');
    const iv = setInterval(async () => {
      try {
        const r = await fetch('/api/recompute-all/status');
        const s = await r.json();
        prog.textContent = `${s.done}/${s.total} — ${s.current || '...'}`;
        if (s.error) prog.title = s.error;
        if (!s.running) {
          clearInterval(iv);
          prog.textContent = '';
          btn.disabled = false;
          this.toast(`Recompute done — ${s.done}/${s.total} images`, 'ok');
          if (Math.random() < 0.4) setTimeout(() => this._marie('✨ Marie, tes nerfs sont magnifiques !'), 1200);
          if (this.cur) this.select(this.cur);
        }
      } catch { clearInterval(iv); prog.textContent = ''; btn.disabled = false; }
    }, 2000);
  }

  /* ── Operation queue ─────────────────────────────────────────────────── */

  /** Push an async operation with an optional display label. */
  _enqueue(fn, label = 'Op') {
    if (!this._qRunning && this._qItems.length === 0) {
      // Fresh batch — reset counters
      this._qTotal = 0;
      this._qDone  = 0;
    }
    const id = ++this._qLogSeq;
    this._qLog.push({ id, label, state: 'pending' });
    if (this._qLog.length > 10) this._qLog.shift();
    this._qItems.push({ fn, id });
    this._qTotal++;
    this._qBar();
    this._renderStack();
    if (!this._qRunning) this._drain();
  }

  async _drain() {
    if (this._qRunning) return;
    this._qRunning = true;
    this._qBar();
    while (this._qItems.length > 0) {
      const { fn, id } = this._qItems.shift();
      const entry = this._qLog.find(e => e.id === id);
      if (entry) { entry.state = 'active'; this._renderStack(); }
      try {
        await fn();
        if (entry) entry.state = 'done';
      } catch {
        if (entry) entry.state = 'err';
      }
      this._qDone++;
      this._renderStack();
      this._qBar();
    }
    this._qRunning = false;
    this._qBarDone();
    setTimeout(() => { this._qLog = []; this._renderStack(); }, 1800);
    if (this._qDirty && this.cur) {
      this._qDirty = false;
      // Regenerate overlay PNG server-side (deferred refresh), then fetch it
      const prefix = this.gtMode ? '/api/gt' : '/api';
      try { await fetch(`${prefix}/image/${this.cur}/rebuild-overlay`, { method: 'POST' }); } catch {}
      this.reloadOverlay();
      if (this.gtMode) this._loadGtList();  // refresh fiber counts
    }
  }

  _renderStack() {
    let el = document.getElementById('action-stack');
    if (!el) {
      el = document.createElement('div');
      el.id = 'action-stack';
      document.getElementById('viewer')?.appendChild(el);
    }
    if (this._qLog.length === 0) { el.innerHTML = ''; return; }
    const pending = this._qLog.filter(e => e.state === 'pending').length;
    const header = pending > 0
      ? `<div class="aq-header">⚙ ${pending} pending</div>`
      : '';
    el.innerHTML = header + this._qLog.map(({ label, state }) => {
      const icon = state === 'active' ? '▶' : state === 'done' ? '✓' : state === 'err' ? '✕' : '·';
      return `<div class="aq-item ${state}">${icon} ${label}</div>`;
    }).join('');
  }

  _moveCursorHint(cx, cy) {
    const el = document.getElementById('cursor-hint');
    if (!el || el.classList.contains('hidden') || this.spaceHeld || this.dragging) return;
    const rc = document.getElementById('viewer').getBoundingClientRect();
    el.style.display = '';
    el.style.left = (cx - rc.left + 16) + 'px';
    el.style.top  = (cy - rc.top  + 16) + 'px';
  }

  _qBar() {
    const total = this._qTotal, done = this._qDone;
    const pct = total > 0 ? Math.round((done / total) * 100) : 0;
    const fill = $('#op-bar-fill');
    if (fill) { fill.style.transition = 'width .1s ease'; fill.style.width = `${pct}%`; fill.style.opacity = '1'; }
    const lbl = $('#queue-status');
    if (!lbl) return;
    const pending = total - done;
    if (pending > 0) {
      lbl.textContent = `⚙ ${pending}`;
      lbl.style.color = 'var(--amber)';
      lbl.classList.remove('hidden');
      $('#btn-recompute').disabled = true;
    }
  }

  _qBarDone() {
    const fill = $('#op-bar-fill');
    const lbl = $('#queue-status');
    if (fill) {
      fill.style.width = '100%';
      setTimeout(() => {
        fill.style.transition = 'width .1s ease, opacity .35s ease';
        fill.style.opacity = '0';
        setTimeout(() => { fill.style.width = '0'; fill.style.opacity = '1'; }, 380);
      }, 220);
    }
    if (lbl) {
      lbl.textContent = '✓';
      lbl.style.color = 'var(--green)';
      setTimeout(() => lbl.classList.add('hidden'), 700);
    }
    $('#btn-recompute').disabled = false;
  }

  _startPresence(stem) {
    clearInterval(this._presenceTimer);
    const ping = () => fetch(`/api/presence/${stem}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: this._clientId }),
    }).then(r => r.json()).then(d => {
      this._presenceCounts[stem] = d.viewers;
      this._updatePresenceBadge(stem, d.viewers);
    }).catch(() => {});
    ping();
    this._presenceTimer = setInterval(ping, 5000);
  }

  _leavePresence(stem) {
    fetch(`/api/presence/${stem}`, {
      method: 'DELETE', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: this._clientId }),
      keepalive: true,   // survives page unload
    }).catch(() => {});
  }

  _stopPresence() {
    clearInterval(this._presenceTimer);
    if (this.cur) this._leavePresence(this.cur);
  }

  async _pollPresence() {
    try {
      const counts = await fetch('/api/presence').then(r => r.json());
      this._presenceCounts = counts;
      // Clear all existing badges first so stale ones (user left) disappear
      document.querySelectorAll('.presence-badge').forEach(el => el.remove());
      for (const [stem, n] of Object.entries(counts)) this._updatePresenceBadge(stem, n);
    } catch {}
  }

  _updatePresenceBadge(stem, n) {
    const li = document.querySelector(`#img-list li[data-stem="${CSS.escape(stem)}"]`);
    if (!li) return;
    let badge = li.querySelector('.presence-badge');
    if (n >= 1) {
      if (!badge) { badge = document.createElement('span'); badge.className = 'presence-badge'; li.insertBefore(badge, li.querySelector('.img-n')); }
      badge.textContent = `👥 ${n}`;
      badge.title = `${n} person${n > 1 ? 's' : ''} viewing this image`;
    } else if (badge) {
      badge.remove();
    }
  }

  _removeAnn(ann) {
    const i = this.anns.indexOf(ann);
    if (i !== -1) this.anns.splice(i, 1);
  }

  _ptCentroid(pts) {
    return {
      x: Math.round(pts.reduce((s, p) => s + p.x, 0) / pts.length),
      y: Math.round(pts.reduce((s, p) => s + p.y, 0) / pts.length),
    };
  }

  /* ── Easter egg 💙 ──────────────────────────────────────────────────── */

  _marie(msg) {
    const el = document.createElement('div');
    el.className = 'toast marie-toast';
    el.textContent = msg;
    $('#toasts').appendChild(el);
    setTimeout(() => el.remove(), 5000);
  }

  /* ── Toast ───────────────────────────────────────────────────────────── */

  toast(msg, type = 'info', ms = 3500) {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = msg;
    $('#toasts').appendChild(el);
    setTimeout(() => el.remove(), ms);
  }

  /** Flash the metrics panel to signal new data arrived. */
  _flashPanel() {
    const p = $('#panel');
    p.classList.add('flash');
    setTimeout(() => p.classList.remove('flash'), 700);
  }
}

document.addEventListener('DOMContentLoaded', () => { window._app = new App(); });
