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
    this.fascPts = [];          // confirmed vertices (image coords)
    this.fascMouse = null;      // live cursor pos for preview
    this._lastFascClick = 0;    // double-click detection
    this.savedFasc = null;      // saved boundary [[x,y],…] from server

    // processed state per stem
    this.processedMap = new Map();

    // annotations (brief markers before overlay refreshes)
    this.anns = [];
    this.editCount = 0;

    // raw image for opacity blending
    this.rawImg = null;
    this.overlayOpacity = 0.85;

    // Easter egg key buffer 💙
    this._keyBuffer = '';
    this._logoClicks = 0;
    this._logoTimer = null;

    this.boot();
  }

  async boot() {
    this.bindEvents();
    this.bindResize();
    await this.loadList();
    if (this.stems.length) {
      this.select(this.stems[0]);
      $('#hint').classList.add('hidden');
    }
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
    $('#btn-recompute').onclick = () => this.recompute();
    $('#btn-reset').onclick = () => this.reset();
    $('#btn-clear-fasc').onclick = () => this.clearFascicle();
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

    this.cvs.addEventListener('mousedown', e => this.onDown(e));
    this.cvs.addEventListener('mousemove', e => this.onMove(e));
    document.addEventListener('mouseup', e => this.onUp(e));
    // Only stop panning on mouseleave — don't cancel drawing
    this.cvs.addEventListener('mouseleave', () => {
      if (this.dragging) {
        this.dragging = false;
        $('#viewer').classList.remove('panning');
      }
      if (this.mode === 'fascicle' && this.fascMouse) {
        this.fascMouse = null;
        this.render();
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
      // Clear polygon when leaving fascicle
      this.fascPts = []; this.fascMouse = null;
    }
    this.mode = m;
    $$('.mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === m));
    $('#viewer').className = `mode-${m}`;
    if (m === 'fiber') {
      this.fiberStep = 0; this.outerPoly = [];
      this.toast('Step 1/2 : draw myelin boundary (outer)', 'info');
    }
    if (m === 'fascicle') {
      this.fascPts = [];
      this.toast('Click to place points — close by clicking the 1st point or double-click', 'info');
    }
    this.render();
  }

  /* ── Image list ──────────────────────────────────────────────────────── */

  async loadList() {
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
      const resegBadge = i.needs_resegment ? `<span class="reseg-badge" title="Edits en attente — cliquer Recompute">⚠</span>` : '';
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

  async select(stem) {
    this.cur = stem;
    $('#current-stem').textContent = stem;
    $$('#img-list li').forEach(li => li.classList.toggle('active', li.dataset.stem === stem));

    this.img = null;
    this.rawImg = null;
    this.anns = [];
    this.editCount = 0;
    this.savedFasc = null;
    this._showFasciclePanel(false);
    this.render();
    $('#loading').classList.remove('hidden');

    const loadImg = src => new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => res(im);
      im.onerror = rej;
      im.src = src;
    });

    (async () => {
      const t = Date.now();
      let im = null;
      let fromRaw = false;
      try {
        im = await loadImg(`/api/image/${stem}/overlay?t=${t}`);
      } catch {
        try {
          im = await loadImg(`/api/image/${stem}/raw?t=${t}`);
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
        // Load raw in background for opacity blending
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
    })();
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
      const r = await fetch(`/api/image/${stem}/fascicle?t=${Date.now()}`);
      if (!r.ok) return;
      const d = await r.json();
      if (this.cur !== stem) return;
      const list = d.contours && d.contours.length > 0 ? d.contours : null;
      this.savedFasc = list && list.length > 0 ? list : null;
      this._showFasciclePanel(!!this.savedFasc);
      this.render();
    } catch (e) { console.warn('loadFascicle:', e); }
  }

  _showFasciclePanel(show) {
    $('#fascicle-panel').classList.toggle('hidden', !show);
  }

  async clearFascicle() {
    if (!this.cur) return;
    try {
      await fetch(`/api/image/${this.cur}/clear-fascicle`, { method: 'POST' });
      this.savedFasc = null;
      this._showFasciclePanel(false);
      if (this.mode === 'fascicle') this.setMode('navigate');
      this.render();
      this.toast('Fascicle cleared', 'info');
    } catch (err) { this.toast(err.message, 'err'); }
  }

  /* ── Metrics panel ───────────────────────────────────────────────────── */

  showMetrics(m) {
    const fmt = (v, d = 4) => v != null ? Number(v).toFixed(d) : '--';
    const rows = [
      ['N axons',            m.n_axons ?? '--',                 true],
      ['Nerve area',         fmt(m.nerve_area_mm2) + ' mm2',    true],
      ['Total axon area',    fmt(m.total_axon_area_mm2) + ' mm2', true],
      ['Total myelin area',  fmt(m.total_myelin_area_mm2) + ' mm2', true],
      ['N-ratio',            fmt(m.nratio),                     true],
      ['G-ratio',            fmt(m.gratio_aggr),                true],
    ];
    const seg = [
      ['AVF',           fmt(m.avf)],
      ['MVF',           fmt(m.mvf)],
      ['Axon density',  fmt(m.axon_density_mm2, 0) + ' /mm2'],
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
    $('#edit-info').textContent = this.editCount
      ? `${this.editCount} edit(s) -- recompute for metrics`
      : '';
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

    // Draw raw first, then overlay with opacity for blending
    if (this.rawImg && this.img && this.img !== this.rawImg) {
      ctx.drawImage(this.rawImg, 0, 0);
      ctx.globalAlpha = this.overlayOpacity;
      ctx.drawImage(this.img, 0, 0);
      ctx.globalAlpha = 1;
    } else if (this.img) {
      ctx.drawImage(this.img, 0, 0);
    }

    // Saved fascicle boundaries — hidden while actively drawing a replacement
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
        ctx.lineWidth = 2.5 / this.zoom;
        ctx.setLineDash([10 / this.zoom, 6 / this.zoom]);
        ctx.shadowBlur = 6;
        ctx.shadowColor = 'rgba(0,0,0,0.8)';
        ctx.stroke();
      }
      // Label above topmost contour
      ctx.setLineDash([]);
      ctx.shadowBlur = 3;
      const top = this.savedFasc[0].reduce((a, b) => a[1] < b[1] ? a : b);
      const label = this.savedFasc.length > 1 ? `FASCICLE ×${this.savedFasc.length}` : 'FASCICLE';
      ctx.font = `bold ${Math.max(11, 13 / this.zoom)}px sans-serif`;
      ctx.fillStyle = '#F1C40F';
      ctx.textAlign = 'center';
      ctx.fillText(label, top[0], top[1] - 8 / this.zoom);
      ctx.restore();
    }

    const r = 14 / this.zoom, lw = 2.5 / this.zoom;

    for (const a of this.anns) {
      ctx.save();
      if (a.t === 'del') {
        ctx.strokeStyle = '#ef5350';
        ctx.lineWidth = lw;
        ctx.beginPath(); ctx.arc(a.x, a.y, r, 0, Math.PI * 2); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(a.x - r * .55, a.y - r * .55);
        ctx.lineTo(a.x + r * .55, a.y + r * .55);
        ctx.moveTo(a.x + r * .55, a.y - r * .55);
        ctx.lineTo(a.x - r * .55, a.y + r * .55);
        ctx.stroke();
      } else if (a.t === 'fiber') {
        // Blue outer ring + green inner + plus
        ctx.lineWidth = lw;
        ctx.strokeStyle = '#2980B9';
        ctx.beginPath(); ctx.arc(a.x, a.y, r * 1.3, 0, Math.PI * 2); ctx.stroke();
        ctx.strokeStyle = '#66bb6a';
        ctx.beginPath(); ctx.arc(a.x, a.y, r * .65, 0, Math.PI * 2); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(a.x, a.y - r * .45); ctx.lineTo(a.x, a.y + r * .45);
        ctx.moveTo(a.x - r * .45, a.y); ctx.lineTo(a.x + r * .45, a.y);
        ctx.stroke();
      } else {
        ctx.strokeStyle = '#66bb6a';
        ctx.lineWidth = lw;
        ctx.beginPath(); ctx.arc(a.x, a.y, r, 0, Math.PI * 2); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(a.x, a.y - r * .55); ctx.lineTo(a.x, a.y + r * .55);
        ctx.moveTo(a.x - r * .55, a.y); ctx.lineTo(a.x + r * .55, a.y);
        ctx.stroke();
      }
      ctx.restore();
    }

    // Stored outer polygon (fiber mode step 2) — blue dashed
    if (this.outerPoly.length > 1) {
      ctx.save();
      ctx.strokeStyle = '#2980B9';
      ctx.lineWidth = 2.5 / this.zoom;
      ctx.setLineDash([6 / this.zoom, 4 / this.zoom]);
      ctx.beginPath();
      ctx.moveTo(this.outerPoly[0].x, this.outerPoly[0].y);
      for (let i = 1; i < this.outerPoly.length; i++)
        ctx.lineTo(this.outerPoly[i].x, this.outerPoly[i].y);
      ctx.closePath();
      ctx.fillStyle = 'rgba(41,128,185,0.12)';
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    // Fiber lasso (blue outer, green inner)
    if (this.drawPts.length > 1) {
      const isOuter = this.mode === 'fiber' && this.fiberStep === 0;
      const clr = isOuter ? '#2980B9' : '#66bb6a';
      const bg  = isOuter ? 'rgba(41,128,185,0.15)' : 'rgba(102,187,106,0.18)';
      ctx.save();
      ctx.strokeStyle = clr;
      ctx.lineWidth = 2 / this.zoom;
      ctx.setLineDash([6 / this.zoom, 4 / this.zoom]);
      ctx.beginPath();
      ctx.moveTo(this.drawPts[0].x, this.drawPts[0].y);
      for (let i = 1; i < this.drawPts.length; i++)
        ctx.lineTo(this.drawPts[i].x, this.drawPts[i].y);
      ctx.closePath();
      ctx.fillStyle = bg;
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    // Fascicle polygon (click-to-add vertices)
    if (this.fascPts.length > 0) {
      const lw = 2 / this.zoom;
      ctx.save();

      // Semi-transparent fill preview
      if (this.fascPts.length >= 3) {
        ctx.beginPath();
        ctx.moveTo(this.fascPts[0].x, this.fascPts[0].y);
        for (let i = 1; i < this.fascPts.length; i++)
          ctx.lineTo(this.fascPts[i].x, this.fascPts[i].y);
        if (this.fascMouse) ctx.lineTo(this.fascMouse.x, this.fascMouse.y);
        ctx.closePath();
        ctx.fillStyle = 'rgba(241,196,15,0.10)';
        ctx.fill();
      }

      // Solid edges between confirmed vertices
      if (this.fascPts.length > 1) {
        ctx.setLineDash([]);
        ctx.strokeStyle = '#F1C40F';
        ctx.lineWidth = lw;
        ctx.beginPath();
        ctx.moveTo(this.fascPts[0].x, this.fascPts[0].y);
        for (let i = 1; i < this.fascPts.length; i++)
          ctx.lineTo(this.fascPts[i].x, this.fascPts[i].y);
        ctx.stroke();
      }

      // Dashed preview edge to cursor
      if (this.fascMouse) {
        const last = this.fascPts[this.fascPts.length - 1];
        ctx.setLineDash([5 / this.zoom, 4 / this.zoom]);
        ctx.strokeStyle = 'rgba(241,196,15,0.7)';
        ctx.lineWidth = lw;
        ctx.beginPath();
        ctx.moveTo(last.x, last.y);
        ctx.lineTo(this.fascMouse.x, this.fascMouse.y);
        ctx.stroke();
      }

      // Vertex dots
      ctx.setLineDash([]);
      for (let i = 0; i < this.fascPts.length; i++) {
        const r = (i === 0 ? 5 : 3) / this.zoom;
        ctx.beginPath();
        ctx.arc(this.fascPts[i].x, this.fascPts[i].y, r, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? '#E67E22' : '#F1C40F';
        ctx.fill();
      }

      // Closing-snap ring around first vertex (3+ pts and mouse close enough)
      if (this.fascPts.length >= 3 && this.fascMouse) {
        const snapR = 14 / this.zoom;
        if (Math.hypot(this.fascMouse.x - this.fascPts[0].x, this.fascMouse.y - this.fascPts[0].y) < snapR) {
          ctx.beginPath();
          ctx.arc(this.fascPts[0].x, this.fascPts[0].y, snapR, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(241,196,15,0.6)';
          ctx.lineWidth = lw;
          ctx.setLineDash([3/this.zoom, 3/this.zoom]);
          ctx.stroke();
        }
      }

      ctx.restore();
    }

    ctx.restore();

    // Magnifying loupe — fascicle mode only, follows cursor (uses raw for precision)
    const loupeImg = this.rawImg || this.img;
    if (this.mode === 'fascicle' && this.fascMouse && loupeImg) {
      const mx = this.fascMouse.x, my = this.fascMouse.y;
      const sX = mx * this.zoom + this.panX;
      const sY = my * this.zoom + this.panY;

      const LR  = 72;   // loupe radius on screen (px)
      const SRC = 52;   // source half-size in image px (fixed detail level)

      // Keep loupe inside canvas — prefer top-right, flip if needed
      let lX = sX + LR + 14;
      let lY = sY - LR - 14;
      if (lX + LR > cw) lX = sX - LR - 14;
      if (lY - LR < 0)  lY = sY + LR + 14;

      // Drop shadow
      ctx.save();
      ctx.shadowBlur = 14;
      ctx.shadowColor = 'rgba(0,0,0,0.65)';
      ctx.beginPath();
      ctx.arc(lX, lY, LR + 1, 0, Math.PI * 2);
      ctx.fillStyle = '#000';
      ctx.fill();
      ctx.shadowBlur = 0;

      // Clip to circle
      ctx.beginPath();
      ctx.arc(lX, lY, LR, 0, Math.PI * 2);
      ctx.clip();

      // Zoomed image (nearest-neighbour for crisp pixels)
      const sx = Math.max(0, mx - SRC);
      const sy = Math.max(0, my - SRC);
      const sw = Math.min(loupeImg.naturalWidth  - sx, SRC * 2);
      const sh = Math.min(loupeImg.naturalHeight - sy, SRC * 2);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(loupeImg, sx, sy, sw, sh, lX - LR, lY - LR, LR * 2, LR * 2);
      ctx.imageSmoothingEnabled = true;

      // Crosshair
      ctx.strokeStyle = 'rgba(241,196,15,0.95)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(lX - 10, lY); ctx.lineTo(lX + 10, lY);
      ctx.moveTo(lX, lY - 10); ctx.lineTo(lX, lY + 10);
      ctx.stroke();
      ctx.restore();

      // Gold border ring
      ctx.beginPath();
      ctx.arc(lX, lY, LR, 0, Math.PI * 2);
      ctx.strokeStyle = '#F1C40F';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    $('#zoom-info').textContent = `${(this.zoom * 100).toFixed(0)}%`;
  }

  s2i(e) {
    const rc = this.cvs.getBoundingClientRect();
    return {
      x: Math.round((e.clientX - rc.left - this.panX) / this.zoom),
      y: Math.round((e.clientY - rc.top - this.panY) / this.zoom),
    };
  }

  /* ── Mouse ───────────────────────────────────────────────────────────── */

  onDown(e) {
    e.preventDefault();
    const isR = e.button === 2, isM = e.button === 1;

    // Pan: right-click, middle-click, space+click, or navigate mode
    if (isR || isM || this.spaceHeld || this.mode === 'navigate') {
      this.dragging = true;
      this.lastM = { x: e.clientX, y: e.clientY };
      $('#viewer').classList.add('panning');
      return;
    }

    if (this.mode === 'delete') this.clickDelete(e);
    if (this.mode === 'fiber') {
      this.drawing = true;
      this.drawPts = [this.s2i(e)];
    }
    if (this.mode === 'fascicle') {
      const pt = this.s2i(e);
      const now = Date.now();
      const dbl = now - this._lastFascClick < 350;
      this._lastFascClick = now;

      if (dbl && this.fascPts.length >= 3) {
        // Double-click → close & submit
        this.submitFascicle([...this.fascPts]);
        this.fascPts = []; this.fascMouse = null;
      } else if (this.fascPts.length >= 3 &&
          Math.hypot(pt.x - this.fascPts[0].x, pt.y - this.fascPts[0].y) < 14 / this.zoom) {
        // Click near first vertex → close & submit
        this.submitFascicle([...this.fascPts]);
        this.fascPts = []; this.fascMouse = null;
      } else {
        this.fascPts.push(pt);
      }
      this.render();
    }
  }

  onMove(e) {
    const pt = this.s2i(e);
    $('#coords').textContent = `x=${pt.x}  y=${pt.y}`;

    if (this.dragging) {
      this.panX += e.clientX - this.lastM.x;
      this.panY += e.clientY - this.lastM.y;
      this.lastM = { x: e.clientX, y: e.clientY };
      this.render();
      return;
    }
    if (this.mode === 'fascicle') {
      this.fascMouse = pt;
      if (this.fascPts.length > 0) this.render();
      return;
    }
    if (this.drawing) {
      this.drawPts.push(pt);
      this.render();
    }
  }

  onUp(e) {
    if (this.dragging) {
      this.dragging = false;
      $('#viewer').classList.remove('panning');
      return;
    }
    if (this.drawing && this.drawPts.length >= 3) {
      if (this.mode === 'fiber') {
        const sampled = this.drawPts.filter((_, i) => i % 3 === 0 || i === this.drawPts.length - 1);
        if (this.fiberStep === 0) {
          // Step 1 done — store outer, move to step 2
          this.outerPoly = sampled;
          this.fiberStep = 1;
          this.drawing = false;
          this.drawPts = [];
          this.render();
          this.toast('Step 2/2 : draw axon inside', 'info');
          return;
        } else {
          // Step 2 done — submit both polygons
          this.submitFiber(sampled);
        }
      }
    }
    this.drawing = false;
    this.drawPts = [];
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

  onKey(e, down) {
    if (e.code === 'Space') { this.spaceHeld = down; if (down) e.preventDefault(); return; }
    if (!down) return;
    // Easter egg — type "marie" anywhere 💙
    if (e.key.length === 1) {
      this._keyBuffer = (this._keyBuffer + e.key.toLowerCase()).slice(-5);
      if (this._keyBuffer === 'marie') { this._keyBuffer = ''; this._marie('💙 Coucou Marie — bonne analyse !'); }
    }
    if (e.key === '1') this.setMode('navigate');
    if (e.key === '2') this.setMode('delete');
    if (e.key === '3') this.setMode('fiber');
    if (e.key === '4') this.setMode('fascicle');
    if (e.key === 'Escape') {
      if (this.mode === 'fiber' && (this.drawing || this.fiberStep > 0)) {
        this.fiberStep = 0; this.outerPoly = []; this.drawPts = [];
        this.drawing = false; this.render(); this.toast('Cancelled', 'info');
      }
      if (this.mode === 'fascicle' && this.fascPts.length > 0) {
        this.fascPts = []; this.fascMouse = null;
        this.render(); this.toast('Cancelled', 'info');
      }
    }
    if (e.key === 'f' || e.key === 'F') this.fit();
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { this.nav(1); e.preventDefault(); }
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { this.nav(-1); e.preventDefault(); }
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
    if (on) {
      el.textContent = msg || 'Processing...';
      el.classList.remove('hidden');
    } else {
      el.classList.add('hidden');
    }
  }

  reloadOverlay() {
    const stem = this.cur;
    if (!stem) return;
    const im = new Image();
    im.onload = () => {
      if (this.cur !== stem) return;
      this.img = im;
      this.anns = [];
      this.showBusy(false);
      this.render();
    };
    im.onerror = () => this.showBusy(false);
    im.src = `/api/image/${stem}/overlay?t=${Date.now()}`;
  }

  async clickDelete(e) {
    if (!this.cur) return;
    const pt = this.s2i(e);
    this.showBusy(true, 'Deleting...');
    try {
      const r = await fetch(`/api/image/${this.cur}/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x: pt.x, y: pt.y }),
      });
      if (!r.ok) {
        this.showBusy(false);
        const err = await r.json();
        this.toast(err.detail || 'No axon here', 'err');
        return;
      }
      const d = await r.json();
      this.anns.push({ t: 'del', x: d.x, y: d.y, label: d.deleted });
      this.editCount++;
      this.showEditCount();
      this.render();
      this.toast(`Deleted axon #${d.deleted}`, 'info');
      if (d.refreshed) this.reloadOverlay(); // hides busy on load
      else this.showBusy(false);
    } catch (err) { this.showBusy(false); this.toast(err.message, 'err'); }
  }

  async submitFiber(innerPts) {
    if (!this.cur) return;
    const outer = this.outerPoly;
    if (outer.length < 3 || innerPts.length < 3) {
      this.toast('Draw larger shapes', 'err'); return;
    }
    this.fiberStep = 0;
    this.outerPoly = [];
    this.showBusy(true, 'Adding fiber...');
    this.render();
    try {
      const r = await fetch(`/api/image/${this.cur}/add-fiber`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          outer_points: outer.map(p => [p.x, p.y]),
          inner_points: innerPts.map(p => [p.x, p.y]),
        }),
      });
      if (!r.ok) {
        this.showBusy(false);
        const err = await r.json();
        this.toast(err.detail || 'Cannot add fiber', 'err');
        return;
      }
      const d = await r.json();
      this.anns.push({ t: 'fiber', x: d.x, y: d.y, label: d.added });
      this.editCount++;
      this.showEditCount();
      this.toast(`Added fiber #${d.added}`, 'ok');
      if (d.refreshed) this.reloadOverlay();
      else this.showBusy(false);
    } catch (err) { this.showBusy(false); this.toast(err.message, 'err'); }
  }

  async submitFascicle(pts) {
    if (!this.cur) return;
    if (pts.length < 3) { this.toast('Draw a larger shape', 'err'); return; }
    this.showBusy(true, 'Saving fascicle...');
    try {
      const r = await fetch(`/api/image/${this.cur}/set-fascicle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ points: pts.map(p => [p.x, p.y]) }),
      });
      if (!r.ok) {
        this.showBusy(false);
        const err = await r.json();
        this.toast(err.detail || 'Failed to save fascicle', 'err');
        return;
      }
      this.showBusy(false);
      this.toast('Fascicle saved — click Recompute to apply', 'ok');
      this.loadFascicle(this.cur);  // server returns accurate union contour
    } catch (err) { this.showBusy(false); this.toast(err.message, 'err'); }
  }

  async recompute() {
    if (!this.cur) return;
    this.toast('Recomputing...', 'info');
    $('#btn-recompute').disabled = true;
    try {
      const r = await fetch(`/api/image/${this.cur}/recompute`, { method: 'POST' });
      if (!r.ok) {
        const err = await r.json();
        this.toast(err.detail || 'Recompute failed', 'err');
        return;
      }
      const d = await r.json();
      this.toast(`Done -- ${d.n_axons} axons`, 'ok');
      this.anns = [];
      this.editCount = 0;
      await this.select(this.cur);
      await this.loadList();
    } catch (err) { this.toast(err.message, 'err'); }
    finally { $('#btn-recompute').disabled = false; }
  }

  async reset() {
    if (!this.cur) return;
    if (!confirm('Reset to original segmentation? All pending edits will be lost.')) return;
    try {
      const r = await fetch(`/api/image/${this.cur}/reset`, { method: 'POST' });
      const d = await r.json();
      if (d.status === 'no_backup') { this.toast('No edits to reset', 'info'); return; }
      this.toast('Reset OK', 'ok');
      this.anns = [];
      this.editCount = 0;
      await this.select(this.cur);
      await this.loadList();
    } catch (err) { this.toast(err.message, 'err'); }
  }

  async recomputeAll() {
    const nProcessed = [...this.processedMap.values()].filter(Boolean).length;
    const minsEst = Math.max(1, Math.round(nProcessed * 1.5));
    const msg =
      `Recomputer ${nProcessed} image(s) ?\n\n` +
      `⏱  Durée estimée : ~${minsEst} min (Cellpose tourne image par image)\n\n` +
      `✅  L'app reste entièrement utilisable pendant le calcul.\n` +
      `⚠️  Les caches Cellpose ne sont PAS relancés — seules les morphométries\n` +
      `    sont recalculées. Pour relancer Cellpose, supprime les caches .npy\n` +
      `    puis lance segment.py.`;
    if (!confirm(msg)) return;
    const btn = $('#btn-recompute-all');
    btn.disabled = true;
    try {
      const r = await fetch('/api/recompute-all', { method: 'POST' });
      const d = await r.json();
      if (d.status === 'already_running') {
        this.toast('Batch recompute already running', 'info');
      } else if (d.status === 'started') {
        this.toast(`Recomputing ${d.total} images in background...`, 'info');
      } else {
        this.toast('No images found', 'info');
        btn.disabled = false;
        return;
      }
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
        prog.textContent = `${s.done}/${s.total} -- ${s.current || '...'}`;
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

  /* ── Easter egg 💙 ──────────────────────────────────────────────────── */

  _marie(msg) {
    const el = document.createElement('div');
    el.className = 'toast marie-toast';
    el.textContent = msg;
    $('#toasts').appendChild(el);
    setTimeout(() => el.remove(), 5000);
  }

  /* ── Toast ───────────────────────────────────────────────────────────── */

  toast(msg, type = 'info') {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = msg;
    $('#toasts').appendChild(el);
    setTimeout(() => el.remove(), 3500);
  }
}

document.addEventListener('DOMContentLoaded', () => new App());
