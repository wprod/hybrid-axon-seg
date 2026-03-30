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

    // fiber mode (two-step: outer then inner)
    this.fiberStep = 0;   // 0 = drawing outer, 1 = drawing inner
    this.outerPoly = [];  // stored after first draw

    // annotations (brief markers before overlay refreshes)
    this.anns = [];
    this.editCount = 0;

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
    $('#btn-recompute-all').onclick = () => this.recomputeAll();
    $('#btn-compare').onclick = () => window.open('/api/comparison', '_blank');
    $('#btn-prev').onclick = () => this.nav(-1);
    $('#btn-next').onclick = () => this.nav(1);

    $$('.view-btn').forEach(b => b.onclick = () => {
      if (this.cur) window.open(`/api/image/${this.cur}/${b.dataset.view}`, '_blank');
    });

    this.cvs.addEventListener('mousedown', e => this.onDown(e));
    this.cvs.addEventListener('mousemove', e => this.onMove(e));
    document.addEventListener('mouseup', e => this.onUp(e));
    // Only stop panning on mouseleave — don't cancel drawing
    this.cvs.addEventListener('mouseleave', () => {
      if (this.dragging) {
        this.dragging = false;
        $('#viewer').classList.remove('panning');
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
    // Reset fiber flow when leaving fiber mode
    if (this.mode === 'fiber' && m !== 'fiber') {
      this.fiberStep = 0;
      this.outerPoly = [];
    }
    this.mode = m;
    $$('.mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === m));
    $('#viewer').className = `mode-${m}`;
    if (m === 'fiber') {
      this.fiberStep = 0;
      this.outerPoly = [];
      this.toast('Step 1/2 : draw myelin boundary (outer)', 'info');
    }
  }

  /* ── Image list ──────────────────────────────────────────────────────── */

  async loadList() {
    const r = await fetch('/api/images');
    const imgs = await r.json();
    this.stems = imgs.map(i => i.stem);
    const ul = $('#img-list');
    ul.innerHTML = '';
    for (const i of imgs) {
      const li = document.createElement('li');
      li.dataset.stem = i.stem;
      li.innerHTML =
        `<span class="dot ${i.modified ? 'mod' : 'orig'}"></span>` +
        `<span class="img-name" title="${i.stem}">${i.stem}</span>` +
        `<span class="img-n">n=${i.n_axons}</span>`;
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
    this.anns = [];
    this.editCount = 0;
    this.render();
    $('#loading').classList.remove('hidden');

    const im = new Image();
    im.onload = () => {
      this.img = im;
      $('#loading').classList.add('hidden');
      $('#hint').classList.add('hidden');
      this.fit();
      this.loadInfo(stem);
    };
    im.onerror = () => {
      $('#loading').classList.add('hidden');
      this.toast('Failed to load overlay', 'err');
    };
    im.src = `/api/image/${stem}/overlay?t=${Date.now()}`;
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

    if (this.img) ctx.drawImage(this.img, 0, 0);

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

    // In-progress drawing — blue when drawing outer, green when drawing inner/axon
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

    ctx.restore();
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
    if (e.key === '1') this.setMode('navigate');
    if (e.key === '2') this.setMode('delete');
    if (e.key === '3') this.setMode('fiber');
    if (e.key === 'Escape' && this.mode === 'fiber') {
      this.fiberStep = 0; this.outerPoly = []; this.drawPts = [];
      this.drawing = false; this.render(); this.toast('Cancelled', 'info');
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
    if (!confirm('Recompute all images? This may take a while but you can keep editing.')) return;
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
          this.toast(`Recompute done -- ${s.done}/${s.total} images`, 'ok');
          if (this.cur) this.select(this.cur);
        }
      } catch { clearInterval(iv); prog.textContent = ''; btn.disabled = false; }
    }, 2000);
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
