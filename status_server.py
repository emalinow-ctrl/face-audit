#!/usr/bin/env python3
"""Live pipeline dashboard — stdlib only, near-zero overhead.

    python status_server.py --config tests/test_config.yaml --host 100.107.85.14 --port 8472

Shows, refreshed every 2.5s in the browser:
  - overall + per-video progress bars (frame / total)
  - current task (which video is actively processing)
  - rolling face-confidence scores from the detector
  - latest flagged frame thumbnail
  - review queue counts

Data comes from <output_dir>/status/*.json (written by main.py at each
checkpoint — one tiny write per ~90 frames), falling back to the checkpoint
SQLite DB when status files don't exist yet. The server itself only reads
small files on each poll; it does no video decoding.

Bind --host to the machine's Tailscale IP so the dashboard is reachable over
the tailnet but not the LAN. Windows Firewall may need an inbound rule for
the port (run once, elevated PowerShell):
  New-NetFirewallRule -DisplayName "face-audit dashboard" -Direction Inbound `
    -LocalPort 8472 -Protocol TCP -Action Allow
"""
from __future__ import annotations

import json
import mimetypes
import sqlite3
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import typer
import yaml

app = typer.Typer(add_completion=False)

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>face-audit live</title>
<style>
body{background:#101014;color:#e8e8e8;font-family:system-ui,sans-serif;margin:0;padding:20px}
#wrap{max-width:1000px;margin:0 auto}
h2{margin:0 0 4px} .sub{color:#888;font-size:13px;margin-bottom:18px}
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
</style></head><body><div id="wrap">
<div class="row"><h2>face-audit live</h2><span id="task" class="meta"></span></div>
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
</div><script>
async function tick(){
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
    const kc=document.getElementById('kconf');
    kc.textContent=s.avg_conf!=null?(s.avg_conf*100).toFixed(0)+'%':'–';
    kc.className='v '+(s.avg_conf==null?'':s.avg_conf>=0.8?'conf good':s.avg_conf>=0.5?'conf mid':'conf low');
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
        <div class="meta">${v.pct.toFixed(1)}%${v.stale!=''?' · '+esc(v.stale):''}</div>`;
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
  setTimeout(tick,2500);
}
function esc(x){return String(x).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
tick();
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
    """Header-only frame count via cv2 if available; None otherwise."""
    try:
        import cv2  # local import: server stays stdlib-only without it
        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return n or None
    except Exception:  # noqa: BLE001 — cv2 missing or unreadable
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
    # frame_00000123_t45.67s.jpg
    try:
        stem = p.stem
        frame = int(stem.split("_")[1])
        t = stem.split("_t")[1].rstrip("s") + "s"
    except (IndexError, ValueError):
        frame, t = 0, "?"
    return {"video": p.parent.name, "frame": frame, "t": t,
            "mtime": int(m), "path": p}


class Ctx:
    out_dir: Path
    ckpt_db: Path
    review_db: Path
    input_dir: Path
    frame_cache: dict = {}


class Handler(BaseHTTPRequestHandler):
    ctx: Ctx

    def log_message(self, *a):  # quiet
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

    def _status_payload(self) -> dict:
        c = self.ctx
        now = datetime.now(timezone.utc)
        videos: dict[str, dict] = {}

        # 1) rich status files written by the pipeline (preferred)
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

        # 2) checkpoint DB fallback (covers runs from before status files existed)
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
        flagged = 0
        try:
            import csv
            rp = c.out_dir / "report.csv"
            if rp.exists():
                with open(rp, newline="") as fh:
                    flagged = sum(1 for _ in csv.DictReader(fh))
        except OSError:
            pass

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
            "flagged_frames": flagged,
            "review": _review_stats(c.review_db),
            "latest_frame": ({k: lf[k] for k in ("video", "frame", "t", "mtime")}
                             if lf else None),
            "as_of": now.strftime("%H:%M:%S"),
        }

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "":
            return self._send(200, HTML.encode(), "text/html; charset=utf-8")
        if path == "/api/status":
            try:
                return self._json(self._status_payload())
            except Exception as e:  # noqa: BLE001 — never 500 the dashboard
                return self._send(500, json.dumps({"error": str(e)}).encode(),
                                   "application/json")
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
        return self._send(404, b"not found", "text/plain")


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
