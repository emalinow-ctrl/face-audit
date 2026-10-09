#!/usr/bin/env python3
"""Local review UI for the manual queue — zero dependencies (stdlib only).

    python review_server.py --config config.yaml

Then open http://localhost:8471 in a browser. Keyboard shortcuts:
    F = contains a face      N = not a face        S = skip
    B = blurred              H = sharp (unblurred)

Labels are stored in the review SQLite DB; `learn.py tune` turns them into
better thresholds / a learned face-confidence mapping.
"""
from __future__ import annotations

import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import typer
import yaml

from review_store import ReviewDB

app = typer.Typer(add_completion=False)

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Veilaudit review</title>
<style>
body{background:#141414;color:#e8e8e8;font-family:system-ui,sans-serif;margin:0;padding:20px}
#wrap{max-width:1100px;margin:0 auto}
#bar{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
#prog{font-size:14px;color:#999}
.card{display:flex;gap:20px;background:#1e1e1e;border:1px solid #333;border-radius:10px;padding:20px}
#crop{max-width:420px;max-height:420px;border:2px solid #555;border-radius:6px;image-rendering:auto}
#frame{max-width:100%;border:1px solid #444;border-radius:6px;margin-top:12px}
.meta{font-size:13px;line-height:1.7;min-width:280px}
.meta b{color:#ffb3b3}
.tag{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;margin-bottom:8px}
.low{background:#5a3b00;color:#ffcf7a}.bord{background:#4a2b00;color:#ffb37a}
.btns{margin-top:16px;display:flex;gap:10px;flex-wrap:wrap}
button{background:#2c2c2c;color:#eee;border:1px solid #555;border-radius:8px;
       padding:12px 22px;font-size:15px;cursor:pointer}
button:hover{background:#3a3a3a;border-color:#888}
button.face{background:#123f1a;border-color:#2c7a35}button.face:hover{background:#175422}
button.notface{background:#471414;border-color:#8a2b2b}button.notface:hover{background:#5c1a1a}
button.blur{background:#3a2a10;border-color:#8a6a2b}button.sharp{background:#471414;border-color:#8a2b2b}
kbd{background:#333;border:1px solid #555;border-radius:4px;padding:1px 6px;font-size:12px}
#done{text-align:center;padding:80px;font-size:20px;color:#8f8}
.hidden{display:none}
</style></head>
<body><div id="wrap">
<div id="bar"><h2 style="margin:0">Veilaudit review queue</h2><div id="prog"></div></div>
<div id="empty" class="hidden"><div id="done">Queue is empty — nothing to review.</div></div>
<div id="card" class="card hidden">
  <div><img id="crop" src=""><img id="frame" src=""></div>
  <div class="meta">
    <div id="reason"></div>
    <div id="info"></div>
    <div class="btns" id="step1">
      <button class="face" onclick="sayFace(1)">Contains a face <kbd>F</kbd></button>
      <button class="notface" onclick="sendLabel(0,null)">Not a face <kbd>N</kbd></button>
      <button onclick="next()">Skip <kbd>S</kbd></button>
    </div>
    <div class="btns hidden" id="step2">
      <div style="width:100%;color:#999;font-size:13px">The face is…</div>
      <button class="blur" onclick="sendLabel(1,'blurred')">Blurred <kbd>B</kbd></button>
      <button class="sharp" onclick="sendLabel(1,'sharp')">Sharp / unblurred <kbd>H</kbd></button>
      <button onclick="sendLabel(1,null)">Unsure <kbd>U</kbd></button>
    </div>
  </div>
</div></div>
<script>
let items=[], idx=0, total=0;
async function refresh(){
  const q = await (await fetch('/api/queue')).json();
  items = q.items; total = q.total; idx = 0; render(); prog();
}
async function prog(){
  const s = await (await fetch('/api/stats')).json();
  document.getElementById('prog').textContent =
    `${s.reviewed} reviewed · ${s.pending} pending`;
}
function render(){
  const card=document.getElementById('card'), empty=document.getElementById('empty');
  if(idx>=items.length){card.classList.add('hidden');empty.classList.remove('hidden');prog();return;}
  card.classList.remove('hidden');empty.classList.add('hidden');
  const it=items[idx];
  document.getElementById('crop').src='/img/crop/'+it.id;
  document.getElementById('frame').src='/img/frame/'+it.id;
  const rc = it.reason==='low_face_conf'?'low':'bord';
  const rt = it.reason==='low_face_conf'?'low face confidence — is this a face?':'borderline blur — confirm the guess';
  document.getElementById('reason').innerHTML=`<span class="tag ${rc}">${rt}</span>`;
  document.getElementById('info').innerHTML=
    `<b>${it.video}</b> · frame ${it.frame} · ${it.timestamp_s.toFixed(2)}s<br>`+
    `face confidence: <b>${(it.face_confidence*100).toFixed(1)}%</b> (det ${(it.det_score*100).toFixed(0)}%)<br>`+
    (it.track_id?`track #${it.track_id}<br>`:'')+
    (it.pipeline_blur?`pipeline guess: <b>${it.pipeline_blur}</b><br>`:'')+
    (it.lap_var!=null?`lap ${it.lap_var.toFixed(0)} · hf ${it.hf_ratio.toFixed(3)} · edge ${it.edge_den.toFixed(3)}`:'');
  document.getElementById('step1').classList.remove('hidden');
  document.getElementById('step2').classList.add('hidden');
}
function sayFace(v){document.getElementById('step1').classList.add('hidden');
  document.getElementById('step2').classList.remove('hidden');}
async function sendLabel(face,blur){
  await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id:items[idx].id,face:face,blur:blur})});
  idx++; render(); prog();
}
function next(){idx++;render();}
document.addEventListener('keydown',e=>{
  const s2=!document.getElementById('step2').classList.contains('hidden');
  const k=e.key.toLowerCase();
  if(!s2){if(k==='f')sayFace(1);else if(k==='n')sendLabel(0,null);else if(k==='s')next();}
  else{if(k==='b')sendLabel(1,'blurred');else if(k==='h')sendLabel(1,'sharp');else if(k==='u')sendLabel(1,null);}
});
refresh();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    db: ReviewDB
    crops_root: Path

    def log_message(self, *a):  # quieter logs
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _serve_image(self, kind: str, item_id: int):
        item = self.db.get(item_id)
        if not item:
            return self._send(404, b"no such item", "text/plain")
        p = Path(item["crop_path"] if kind == "crop" else item["frame_path"]).resolve()
        if self.crops_root not in p.parents and p.parent != self.crops_root:
            return self._send(403, b"forbidden", "text/plain")
        if not p.exists():
            return self._send(404, b"missing file", "text/plain")
        ctype = mimetypes.guess_type(str(p))[0] or "image/jpeg"
        self._send(200, p.read_bytes(), ctype)

    def do_GET(self):
        parts = urlparse(self.path).path.strip("/").split("/")
        if self.path == "/" or self.path == "":
            return self._send(200, HTML.encode(), "text/html; charset=utf-8")
        if parts == ["api", "queue"]:
            items = self.db.pending(limit=500)
            return self._json({"items": items, "total": len(items)})
        if parts == ["api", "stats"]:
            return self._json(self.db.stats())
        if len(parts) == 3 and parts[0] == "img" and parts[1] in ("crop", "frame"):
            try:
                return self._serve_image(parts[1], int(parts[2]))
            except ValueError:
                return self._send(400, b"bad id", "text/plain")
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/api/label":
            return self._send(404, b"not found", "text/plain")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            item_id = int(body["id"])
            face = body.get("face")
            blur = body.get("blur")
            assert face in (0, 1, None) and blur in ("blurred", "sharp", None)
        except (ValueError, AssertionError, KeyError, TypeError):
            return self._send(400, b"bad request", "text/plain")
        if not self.db.get(item_id):
            return self._send(404, b"no such item", "text/plain")
        self.db.set_label(item_id, face, blur)
        self._json({"ok": True, "stats": self.db.stats()})


@app.command()
def serve(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    port: int | None = typer.Option(None, "--port", "-p"),
):
    cfg = yaml.safe_load(config.read_text())
    rq = cfg.get("review_queue", {})
    Handler.db = ReviewDB(rq.get("db", "review.sqlite"))
    Handler.crops_root = Path(rq.get("crops_dir", "review")).resolve()
    port = port or int(rq.get("server_port", 8471))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    n = Handler.db.stats()["pending"]
    print(f"Review {n} pending items at http://127.0.0.1:{port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    sys.exit(app())
