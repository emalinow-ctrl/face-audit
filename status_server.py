#!/usr/bin/env python3
"""Live pipeline dashboard — stdlib only, near-zero overhead.

    python status_server.py --config tests/test_config.yaml --host 100.107.85.14 --port 8472

Tabs:
  Live — progress bars, current task, rolling face-confidence, latest frame.
  Flagged Frames — gallery of all flagged frames from report.csv with a
    Tinder-style accept/deny reviewer. Verdicts are stored in
    <output_dir>/verdicts.json and can feed the learning loop later.

Data comes from <output_dir>/status/*.json (written by main.py at each
checkpoint), falling back to the checkpoint SQLite DB. The server only reads
small files on each poll; it does no video decoding.

Bind --host to the machine's Tailscale IP so the dashboard is reachable over
the tailnet but not the LAN. Windows Firewall may need an inbound rule for
the port (run once, elevated PowerShell):
  New-NetFirewallRule -DisplayName "Veilaudit dashboard" -Direction Inbound `
    -LocalPort 8472 -Protocol TCP -Action Allow
"""
from __future__ import annotations

import csv
import json
import mimetypes
import sqlite3
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import typer
import yaml

app = typer.Typer(add_completion=False)

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Veilaudit</title>
<style>
body{background:#101014;color:#e8e8e8;font-family:system-ui,sans-serif;margin:0;padding:20px}
#wrap{max-width:1000px;margin:0 auto}
h2{margin:0 0 4px} .sub{color:#888;font-size:13px;margin-bottom:18px}
.hidden{display:none!important}
/* tabs */
#tabs{display:flex;gap:8px;margin:14px 0 18px}
.tab{background:#1c1c22;color:#aaa;border:1px solid #2c2c34;border-radius:10px;
  padding:10px 18px;font-size:14px;cursor:pointer}
.tab.active{background:#2a1545;color:#e8d5ff;border-color:#7a2bd8}
.tab .n{background:#7a2bd8;color:#fff;border-radius:10px;padding:1px 8px;font-size:12px;margin-left:6px}
/* live tab */
.bar{height:22px;background:#222;border-radius:11px;overflow:hidden;margin:6px 0 2px}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,#7a2bd8,#b04dff);
  border-radius:11px;transition:width .8s}
.bar.big{height:30px;border-radius:15px}.bar.big>i{border-radius:15px}
.card{background:#18181e;border:1px solid #2c2c34;border-radius:10px;padding:14px 16px;margin:12px 0}
.card.active{border-color:#7a2bd8}
.row{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
.name{font-weight:600}.meta{font-size:12px;color:#999}
.conf{font-size:13px;font-weight:700}
.conf.good{color:#7ddf8a}.conf.mid{color:#ffcf7a}.conf.low{color:#ff8a8a}
.pill{display:inline-block;padding:2px 10px;border-radius:10px;font-size:11px;margin-left:8px}
.pill.run{background:#2a1545;color:#d9a7ff}.pill.done{background:#123f1a;color:#7ddf8a}
.pill.idle{background:#2a2a2a;color:#999}
#frame{max-width:100%;border:1px solid #333;border-radius:8px;margin-top:8px}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:12px 0}
.kv>div{background:#18181e;border:1px solid #2c2c34;border-radius:10px;padding:10px 14px}
.kv .v{font-size:22px;font-weight:700}.kv .k{font-size:11px;color:#888}
#err{color:#ff8a8a;font-size:13px}
/* flagged tab */
#gallery{margin-top:12px}
.video-section{margin-bottom:24px}
.video-header{font-size:15px;font-weight:600;margin-bottom:10px;padding-bottom:6px;
  border-bottom:1px solid #2c2c34;cursor:pointer;user-select:none}
.video-header .arrow{display:inline-block;width:20px;color:#7a2bd8}
.video-grid.collapsed{display:none}
.video-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}
.thumb{position:relative;cursor:pointer;border-radius:8px;overflow:hidden;border:2px solid #2c2c34}
.thumb img{width:100%;height:110px;object-fit:cover;display:block}
.thumb .cap{font-size:11px;color:#aaa;padding:4px 6px;background:#18181e}
.thumb .badge{position:absolute;top:6px;right:6px;font-size:16px;background:rgba(0,0,0,.6);
  border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center}
.thumb.done-accept{border-color:#2c7a35}.thumb.done-deny{border-color:#8a2b2b;opacity:.55}
/* reviewer */
#reviewer{position:fixed;inset:0;background:rgba(8,8,10,.96);z-index:50;
  display:flex;flex-direction:column;align-items:center;justify-content:center;padding:16px}
#rev-img{max-width:92vw;max-height:62vh;border-radius:12px;border:2px solid #444;touch-action:pan-y}
#rev-meta{margin:12px 0 4px;font-size:14px;color:#ccc;text-align:center}
#rev-count{font-size:12px;color:#888;margin-bottom:10px}
.swipe-btns{display:flex;gap:36px;margin-top:8px}
.sbtn{width:84px;height:84px;border-radius:50%;font-size:36px;cursor:pointer;border:3px solid;
  display:flex;align-items:center;justify-content:center;background:#1c1c22}
.sbtn.deny{border-color:#8a2b2b;color:#ff8a8a}.sbtn.deny:active{background:#471414}
.sbtn.accept{border-color:#2c7a35;color:#7ddf8a}.sbtn.accept:active{background:#123f1a}
#rev-close{position:absolute;top:14px;right:18px;background:none;border:none;color:#888;
  font-size:28px;cursor:pointer}
#rev-empty{color:#888;font-size:16px;padding:40px;text-align:center}
</style></head><body><div id="wrap">
<div class="row"><h2>Veilaudit</h2><span id="task" class="meta"></span></div>
<div id="tabs">
  <button class="tab active" id="tab-live" onclick="showTab('live')">Live</button>
  <button class="tab" id="tab-flagged" onclick="showTab('flagged')">Flagged Frames<span class="n" id="flag-n">0</span></button>
  <button class="tab" id="tab-opencode" onclick="showTab('opencode')">OpenCode<span class="n hidden" id="oc-n"></span></button>
</div>

<div id="page-live">
<div class="sub">refreshes every 2.5s · <span id="asof"></span></div>
<div id="err"></div>
<div class="bar big"><i id="obar" style="width:0%"></i></div>
<div class="meta" id="opct" style="margin-bottom:6px"></div>
<div class="kv">
  <div><div class="v" id="kflag">–</div><div class="k">flagged frames</div></div>
  <div><div class="v" id="kfaces">–</div><div class="k">faces failed</div></div>
  <div><div class="v" id="krev">–</div><div class="k">review pending</div></div>
  <div><div class="v" id="kconf">–</div><div class="k">avg face confidence</div></div>
</div>
<div id="videos"></div>
<div class="card"><div class="row"><span class="name">latest flagged frame</span>
<span class="meta" id="frmeta"></span></div><img id="frame" src=""></div>
</div>

<div id="page-flagged" class="hidden">
<div class="sub" id="flag-stat"></div>
<div id="gallery"></div>
</div>
</div>

<div id="reviewer" class="hidden">
<button id="rev-close" onclick="closeReview()">×</button>
<img id="rev-img" src="">
<div id="rev-meta"></div>
<div id="rev-count"></div>
<div class="swipe-btns">
  <button class="sbtn deny" onclick="verdict('deny')" title="False positive (←)">✕</button>
  <button class="sbtn accept" onclick="verdict('accept')" title="Correct flag (→)">♥</button>
</div>
<div class="meta" style="margin-top:10px">← deny · → accept · swipe or tap</div>
</div>

<div id="page-opencode" class="hidden">
  <div class="card">
    <div class="row"><span class="name">OpenCode Optimization</span><span class="pill" id="oc-status">idle</span></div>
    <div class="kv">
      <div><div class="v" id="oc-phase">-</div><div class="k">Phase</div></div>
      <div><div class="v" id="oc-task">-</div><div class="k">Current Task</div></div>
      <div><div class="v" id="oc-elapsed">-</div><div class="k">Elapsed</div></div>
      <div><div class="v" id="oc-updated">-</div><div class="k">Last Update</div></div>
    </div>
  </div>
  <div class="card">
    <div class="name">Recent Log</div>
    <pre id="oc-log" style="background:#0a0a0c;padding:12px;border-radius:8px;overflow-x:auto;font-size:12px;max-height:300px;overflow-y:auto">No updates yet. POST to /api/opencode to report progress.</pre>
  </div>
  <div class="card">
    <div class="name">Files Changed</div>
    <div id="oc-files" class="meta">None yet.</div>
  </div>
</div>
<script>
let flagged=[], revIdx=-1;
/* ---------- tabs ---------- */
function showTab(which){
  document.getElementById('tab-live').classList.toggle('active', which==='live');
  document.getElementById('tab-flagged').classList.toggle('active', which==='flagged');
  document.getElementById('tab-opencode').classList.toggle('active', which==='opencode');
  document.getElementById('page-live').classList.toggle('hidden', which!=='live');
  document.getElementById('page-flagged').classList.toggle('hidden', which!=='flagged');
  document.getElementById('page-opencode').classList.toggle('hidden', which!=='opencode');
  if(which==='flagged') loadFlagged();
  if(which==='opencode') loadOpencode();
}
/* ---------- live ---------- */
async function tick(){
  if(!document.getElementById('page-live').classList.contains('hidden')){
    try{
      const s = await (await fetch('/api/status')).json();
      document.getElementById('err').textContent='';
      document.getElementById('obar').style.width = s.overall_pct.toFixed(1)+'%';
      document.getElementById('opct').textContent =
        s.overall_pct.toFixed(1)+'% overall · '+s.done_videos+'/'+s.total_videos+' videos done';
      document.getElementById('task').innerHTML = s.pipeline_active
        ? '<span class="pill run">RUNNING</span> '+esc(s.current_task)
        : '<span class="pill idle">IDLE</span> '+(s.current_task?esc(s.current_task):'no active run');
      document.getElementById('asof').textContent='updated '+s.as_of;
      document.getElementById('kflag').textContent=s.flagged_frames;
      document.getElementById('kfaces').textContent=s.faces_failed;
      document.getElementById('krev').textContent=s.review.pending;
      document.getElementById('flag-n').textContent=s.flagged_frames;
      const kc=document.getElementById('kconf');
      kc.textContent=s.avg_conf!=null?(s.avg_conf*100).toFixed(0)+'%':'–';
      const vv=document.getElementById('videos'); vv.innerHTML='';
      for(const v of s.videos){
        const d=document.createElement('div');
        d.className='card'+(v.active?' active':'');
        const pill=v.status==='done'?'<span class="pill done">done</span>'
          :v.active?'<span class="pill run">processing</span>':'<span class="pill idle">waiting</span>';
        const conf=v.avg_conf!=null
          ?`<span class="conf ${v.avg_conf>=0.8?'good':v.avg_conf>=0.5?'mid':'low'}">${(v.avg_conf*100).toFixed(0)}%</span>`:'–';
        d.innerHTML=`<div class="row"><span class="name">${esc(v.name)}${pill}</span>
          <span class="meta">frame ${v.frame.toLocaleString()}${v.total?' / '+v.total.toLocaleString():''}
          · conf ${conf} · ${v.faces_failed} failed · ${v.review_queued} review</span></div>
          <div class="bar"><i style="width:${v.pct.toFixed(1)}%"></i></div>
          <div class="meta">${v.pct.toFixed(1)}%${v.stale?' · '+esc(v.stale):''}</div>`;
        vv.appendChild(d);
      }
      if(s.latest_frame){
        document.getElementById('frmeta').textContent=
          s.latest_frame.video+' · frame '+s.latest_frame.frame+' · '+s.latest_frame.t;
        const im=document.getElementById('frame');
        const url='/api/frame?m='+s.latest_frame.mtime;
        if(im.getAttribute('src')!==url) im.src=url;
      }
    }catch(e){document.getElementById('err').textContent='lost contact with server — retrying…';}
  }
  setTimeout(tick,2500);
}
/* ---------- flagged gallery ---------- */
async function loadFlagged(){
  const d = await (await fetch('/api/flagged')).json();
  flagged = d.frames;
  const unrev = flagged.filter(f=>!f.verdict).length;
  document.getElementById('flag-n').textContent = flagged.length;
  document.getElementById('flag-stat').textContent =
    flagged.length===0 ? 'No flagged frames yet — run the pipeline first.'
    : `${flagged.length} images flagged · ${unrev} ready to review` +
      (unrev===0 ? ' — all reviewed ✓' : ' — tap any frame to start swiping');
  // group frames by video
  const byVideo={};
  flagged.forEach((f,i)=>{ f._idx=i; (byVideo[f.video]=byVideo[f.video]||[]).push(f); });
  const g = document.getElementById('gallery'); g.innerHTML='';
  for(const video of Object.keys(byVideo).sort()){
    const frames=byVideo[video];
    const vUnrev=frames.filter(f=>!f.verdict).length;
    const sec=document.createElement('div');
    sec.className='video-section';
    sec.innerHTML=`<div class="video-header"><span class="arrow">▼</span>${esc(video)} <span class="meta">`+
      `${frames.length} frames · ${vUnrev} to review</span></div>`;
    const grid=document.createElement('div');
    grid.className='video-grid';
    const header=sec.querySelector('.video-header');
    header.onclick=()=>{
      const collapsed=grid.classList.toggle('collapsed');
      header.querySelector('.arrow').textContent=collapsed?'▶':'▼';
    };
    frames.forEach(f=>{
      const el=document.createElement('div');
      el.className='thumb'+(f.verdict==='accept'?' done-accept':f.verdict==='deny'?' done-deny':'');
      el.innerHTML=`${f.verdict?`<div class="badge">${f.verdict==='accept'?'♥':'✕'}</div>`:''}`+
        `<img loading="lazy" src="/api/flagged_img?img=${encodeURIComponent(f.image)}">`+
        `<div class="cap">${esc(f.t)} · ${f.faces_failed} face(s)</div>`;
      el.onclick=()=>openReview(f._idx);
      grid.appendChild(el);
    });
    sec.appendChild(grid);
    g.appendChild(sec);
  }
}
/* ---------- tinder reviewer ---------- */
function openReview(i){
  revIdx=i; renderReview();
  document.getElementById('reviewer').classList.remove('hidden');
}
function closeReview(){
  document.getElementById('reviewer').classList.add('hidden');
  loadFlagged();
}
function renderReview(){
  const f=flagged[revIdx];
  if(!f){closeReview();return;}
  document.getElementById('rev-img').src='/api/flagged_img?img='+encodeURIComponent(f.image);
  document.getElementById('rev-meta').innerHTML=
    `<b>${esc(f.video)}</b> · frame ${f.frame} · ${esc(f.t)} · ${f.faces_failed} face(s) failed`+
    (f.verdict?` · <span style="color:${f.verdict==='accept'?'#7ddf8a':'#ff8a8a'}">`+
      (f.verdict==='accept'?'♥ accepted':'✕ denied')+'</span>':'');
  const unrev=flagged.filter(x=>!x.verdict).length;
  document.getElementById('rev-count').textContent=`${revIdx+1} of ${flagged.length} · ${unrev} left to review`;
}
async function verdict(v){
  const f=flagged[revIdx];
  if(!f) return;
  await fetch('/api/verdict',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({image:f.image,verdict:v})});
  f.verdict=v;
  // advance to next unreviewed
  let n=revIdx+1;
  while(n<flagged.length && flagged[n].verdict) n++;
  if(n>=flagged.length){
    n=0; while(n<flagged.length && flagged[n].verdict) n++;
  }
  if(n>=flagged.length || flagged.every(x=>x.verdict)){ closeReview(); return; }
  revIdx=n; renderReview();
}
document.addEventListener('keydown',e=>{
  if(document.getElementById('reviewer').classList.contains('hidden')) return;
  if(e.key==='ArrowLeft') verdict('deny');
  else if(e.key==='ArrowRight') verdict('accept');
  else if(e.key==='Escape') closeReview();
});
/* swipe */
let tx0=null;
const ri=document.getElementById('rev-img');
ri.addEventListener('touchstart',e=>{tx0=e.touches[0].clientX;},{passive:true});
ri.addEventListener('touchend',e=>{
  if(tx0===null) return;
  const dx=e.changedTouches[0].clientX-tx0; tx0=null;
  if(Math.abs(dx)>60) verdict(dx>0?'accept':'deny');
},{passive:true});
function esc(x){return String(x).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
tick();
async function loadOpencode(){
  try{
    const s = await (await fetch('/api/opencode')).json();
    const pill = document.getElementById('oc-status');
    pill.textContent = s.status || 'idle';
    pill.className = 'pill ' + (s.status==='running' ? 'run' : s.status==='complete' ? 'done' : 'idle');
    document.getElementById('oc-phase').textContent = s.phase || '-';
    document.getElementById('oc-task').textContent = s.task || '-';
    const el = s.elapsed_sec || 0;
    document.getElementById('oc-elapsed').textContent =
      Math.floor(el/3600)+'h '+Math.floor(el%3600/60)+'m '+Math.floor(el%60)+'s';
    document.getElementById('oc-updated').textContent = s.updated_at ? new Date(s.updated_at).toLocaleTimeString() : '-';
    document.getElementById('oc-log').textContent = (s.log_lines||[]).join('\n') || 'No log lines.';
    const files = s.files_changed||[];
    document.getElementById('oc-files').innerHTML = files.length
      ? files.map(f=>'<div>'+f+'</div>').join('') : 'None yet.';
    const n = document.getElementById('oc-n');
    if(s.status==='running'){ n.textContent='●'; n.classList.remove('hidden'); }
    else n.classList.add('hidden');
  }catch(e){ /* silent */ }
}
setInterval(()=>{ if(!document.getElementById('page-opencode').classList.contains('hidden')) loadOpencode(); }, 5000);
</script></body></html>
"""


def _review_stats(db_path: Path) -> dict:
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        row = con.execute(
            "SELECT COUNT(*) FROM review_items WHERE reviewed_at IS NULL"
        ).fetchone()
        total = con.execute("SELECT COUNT(*) FROM review_items").fetchone()
        con.close()
        return {"pending": row[0] if row else 0,
                "reviewed": (total[0] - row[0]) if total and row else 0}
    except (sqlite3.Error, OSError):
        return {"pending": 0, "reviewed": 0}


def _checkpoint_rows(db_path: Path) -> list[dict]:
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        rows = con.execute(
            "SELECT path, last_frame, status, updated_at FROM videos").fetchall()
        con.close()
        return [{"path": r[0], "frame": r[1], "status": r[2], "updated": r[3]}
                for r in rows]
    except (sqlite3.Error, OSError):
        return []


def _video_total_frames(video_path: Path) -> int | None:
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return n or None
    except Exception:  # noqa: BLE001
        return None


def _latest_frame(out_dir: Path) -> dict | None:
    best = None
    try:
        for vdir in out_dir.iterdir():
            if not vdir.is_dir() or vdir.name == "status":
                continue
            for p in vdir.glob("frame_*.jpg"):
                try:
                    m = p.stat().st_mtime
                except OSError:
                    continue
                if best is None or m > best[1]:
                    best = (p, m)
    except OSError:
        return None
    if not best:
        return None
    p, m = best
    try:
        stem = p.stem
        frame = int(stem.split("_")[1])
        t = stem.split("_t")[1].rstrip("s") + "s"
    except (IndexError, ValueError):
        frame, t = 0, "?"
    return {"video": p.parent.name, "frame": frame, "t": t,
            "mtime": int(m), "path": p}


# ---------- flagged-frame verdicts ----------
def _verdicts_file(out_dir: Path) -> Path:
    return out_dir / "verdicts.json"


def _load_verdicts(out_dir: Path) -> dict:
    try:
        return json.loads(_verdicts_file(out_dir).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_verdict(out_dir: Path, image: str, verdict: str) -> None:
    try:
        d = _load_verdicts(out_dir)
        d[image] = {"verdict": verdict,
                    "at": datetime.now(timezone.utc).isoformat()}
        _verdicts_file(out_dir).write_text(json.dumps(d, indent=1))
    except OSError:
        pass


def _flagged_frames(out_dir: Path) -> list[dict]:
    """Every flagged frame from report.csv, annotated with its verdict."""
    verdicts = _load_verdicts(out_dir)
    frames: list[dict] = []
    try:
        rp = out_dir / "report.csv"
        if not rp.exists():
            return []
        with open(rp, newline="") as fh:
            for r in csv.DictReader(fh):
                img = r.get("image", "")
                frames.append({
                    "image": img,
                    "video": r.get("video", ""),
                    "frame": int(r.get("frame") or 0),
                    "t": r.get("timestamp_hms") or (r.get("timestamp_s", "?") + "s"),
                    "faces_failed": int(r.get("faces_failed") or 0),
                    "verdict": verdicts.get(img, {}).get("verdict"),
                })
    except (OSError, ValueError):
        pass
    frames.sort(key=lambda f: (f["verdict"] is not None, f["video"], f["frame"]))
    return frames


class Ctx:
    out_dir: Path
    ckpt_db: Path
    review_db: Path
    input_dir: Path


def _opencode_status(out_dir: Path) -> dict:
    """Read latest OpenCode progress, or empty state."""
    f = out_dir / "status" / "opencode.json"
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return {"status": "idle", "phase": "", "task": "",
                "elapsed_sec": 0, "log_lines": [], "files_changed": []}


def _save_opencode_status(out_dir: Path, data: dict) -> None:
    """Persist OpenCode progress update."""
    d = out_dir / "status"
    d.mkdir(parents=True, exist_ok=True)
    allowed = {"status", "phase", "task", "elapsed_sec", "log_lines", "files_changed"}
    clean = {k: data[k] for k in allowed if k in data}
    clean["updated_at"] = datetime.now(timezone.utc).isoformat()
    (d / "opencode.json").write_text(json.dumps(clean, indent=2))


class Handler(BaseHTTPRequestHandler):
    ctx: Ctx

    def log_message(self, *a):
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj).encode(), "application/json")

    def _resolve_image(self, rel: str) -> Path | None:
        """Resolve a relative image path under out_dir; None if unsafe/missing."""
        try:
            p = (self.ctx.out_dir / rel).resolve()
        except (OSError, ValueError):
            return None
        if self.ctx.out_dir.resolve() not in p.parents:
            return None
        return p if p.is_file() else None

    def _status_payload(self) -> dict:
        c = self.ctx
        now = datetime.now(timezone.utc)
        videos: dict[str, dict] = {}

        sdir = c.out_dir / "status"
        if sdir.is_dir():
            for p in sdir.glob("*.json"):
                try:
                    d = json.loads(p.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                try:
                    age = (now - datetime.fromisoformat(d["updated_at"])).total_seconds()
                except (KeyError, ValueError):
                    age = 1e9
                st = d.get("stats", {})
                confs = d.get("recent_confidence") or []
                videos[d.get("video", p.stem)] = {
                    "name": d.get("video", p.stem),
                    "frame": int(d.get("frame", 0)),
                    "total": int(d.get("total_frames") or 0),
                    "fps": d.get("fps"),
                    "status": d.get("status", "?"),
                    "active": age < 30 and d.get("status") == "processing",
                    "faces_seen": st.get("faces_seen", 0),
                    "faces_failed": st.get("faces_failed", 0),
                    "review_queued": st.get("review_queued", 0),
                    "tracks": st.get("tracks_total", 0),
                    "avg_conf": (sum(confs) / len(confs)) if confs else None,
                    "age_s": age,
                }

        for r in _checkpoint_rows(c.ckpt_db):
            name = Path(r["path"]).name
            if name in videos:
                continue
            total = _video_total_frames(Path(r["path"]))
            try:
                age = (now - datetime.fromisoformat(r["updated"])).total_seconds()
            except (TypeError, ValueError):
                age = 1e9
            videos[name] = {
                "name": name, "frame": r["frame"], "total": total or 0,
                "fps": None, "status": r["status"],
                "active": age < 60 and r["status"] == "in_progress",
                "faces_seen": 0, "faces_failed": 0, "review_queued": 0,
                "tracks": 0, "avg_conf": None, "age_s": age,
            }

        vids = sorted(videos.values(), key=lambda v: v["name"])
        for v in vids:
            v["pct"] = (100.0 * v["frame"] / v["total"]) if v["total"] else 0.0
            v["stale"] = ("updated %.0fs ago" % v["age_s"]) if v["age_s"] < 1e9 else ""
            del v["age_s"]

        active = [v for v in vids if v["active"]]
        done = [v for v in vids if v["status"] == "done"]
        tot_frames = sum(v["frame"] for v in vids)
        tot_total = sum(v["total"] for v in vids)
        confs = [v["avg_conf"] for v in vids if v["avg_conf"] is not None]
        flagged = _flagged_frames(c.out_dir)

        lf = _latest_frame(c.out_dir)
        return {
            "pipeline_active": bool(active),
            "current_task": (active[0]["name"] + " — frame %d" % active[0]["frame"]
                             if active else ("last run finished" if done else "")),
            "overall_pct": (100.0 * tot_frames / tot_total) if tot_total else 0.0,
            "total_videos": len(vids), "done_videos": len(done),
            "videos": vids,
            "avg_conf": (sum(confs) / len(confs)) if confs else None,
            "faces_failed": sum(v["faces_failed"] for v in vids),
            "flagged_frames": len(flagged),
            "review": _review_stats(c.review_db),
            "latest_frame": ({k: lf[k] for k in ("video", "frame", "t", "mtime")}
                             if lf else None),
            "as_of": now.strftime("%H:%M:%S"),
        }

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "":
            return self._send(200, HTML.encode(), "text/html; charset=utf-8")
        if path == "/api/status":
            try:
                return self._json(self._status_payload())
            except Exception as e:  # noqa: BLE001
                return self._send(500, json.dumps({"error": str(e)}).encode(),
                                   "application/json")
        if path == "/api/flagged":
            frames = _flagged_frames(self.ctx.out_dir)
            return self._json({
                "frames": frames,
                "total": len(frames),
                "reviewed": sum(1 for f in frames if f["verdict"]),
            })
        if path == "/api/flagged_img":
            rel = parse_qs(parsed.query).get("img", [""])[0]
            p = self._resolve_image(rel)
            if not p:
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(str(p))[0] or "image/jpeg"
            try:
                return self._send(200, p.read_bytes(), ctype)
            except OSError:
                return self._send(404, b"missing file", "text/plain")
        if path == "/api/frame":
            lf = _latest_frame(self.ctx.out_dir)
            if not lf:
                return self._send(404, b"no flagged frames yet", "text/plain")
            p = lf["path"].resolve()
            if self.ctx.out_dir.resolve() not in p.parents:
                return self._send(403, b"forbidden", "text/plain")
            ctype = mimetypes.guess_type(str(p))[0] or "image/jpeg"
            try:
                return self._send(200, p.read_bytes(), ctype)
            except OSError:
                return self._send(404, b"missing file", "text/plain")
        if path == "/api/opencode":
            return self._json(_opencode_status(self.ctx.out_dir))
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        ppath = urlparse(self.path).path
        if ppath == "/api/opencode":
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                _save_opencode_status(self.ctx.out_dir, body)
            except (ValueError, KeyError, TypeError):
                return self._send(400, b"bad request", "text/plain")
            return self._json({"ok": True})
        if ppath != "/api/verdict":
            return self._send(404, b"not found", "text/plain")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            image = str(body["image"])
            verdict = str(body["verdict"])
            assert verdict in ("accept", "deny")
            assert self._resolve_image(image) is not None
        except (ValueError, AssertionError, KeyError, TypeError):
            return self._send(400, b"bad request", "text/plain")
        _save_verdict(self.ctx.out_dir, image, verdict)
        frames = _flagged_frames(self.ctx.out_dir)
        self._json({"ok": True, "total": len(frames),
                    "reviewed": sum(1 for f in frames if f["verdict"])})


@app.command()
def serve(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8472, "--port", "-p"),
):
    cfg = yaml.safe_load(config.read_text())
    c = Ctx()
    c.out_dir = Path(cfg["paths"]["output_dir"])
    c.ckpt_db = Path(cfg["paths"]["checkpoint_db"])
    c.review_db = Path(cfg.get("review_queue", {}).get("db", "review.sqlite"))
    c.input_dir = Path(cfg["paths"]["input_dir"])
    Handler.ctx = c
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"dashboard at http://{host}:{port}  (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    sys.exit(app())
