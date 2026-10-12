"""Educational playground for the XSS-detection experiment (local, teaching only).

This is NOT the test target. app/vuln_app.py stays deliberately bare so the experiment
measures the detector, not a UI. This layer sits on top and makes the experiment visible:

  type an input  ->  see what the DETECTOR sees + its score/verdict
                 ->  see what the BROWSER does (rendered as text, or marker executed)
                 ->  side by side: same input, safe endpoint vs unsafe endpoint

The lesson the whole project turns on: the detector sees an identical request in both
columns, yet the browser behaves completely differently. Detection != execution.

Runs on http://127.0.0.1:5001 . Needs the vulnerable app on :5000 and models/baseline.joblib.
Everything is localhost; the only "attack" is the harmless marker window.__xss_fired=true.
"""
import sys
import urllib.request
from pathlib import Path
from urllib.parse import quote

import joblib
from flask import Flask, Response, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import APP_URL, MODELS, SAFE_ENDPOINT, UNSAFE_ENDPOINTS, decode_once, norm  # noqa: E402

app = Flask(__name__)
ENDPOINTS = [SAFE_ENDPOINT] + UNSAFE_ENDPOINTS

bundle = joblib.load(MODELS / "baseline.joblib")
MODEL, THRESH = bundle["pipeline"], bundle["threshold"]

# (group, label, value) — grouped so the teaching order is obvious in the UI
PRESETS = [
    ("benign", "Math and code text", "if (a < b && c > d) return a;"),
    ("benign", "HTML inside a question", "how do I close a <div> tag?"),
    ("attack", "Image onerror", "<img src=x onerror=window.__xss_fired=true>"),
    ("attack", "SVG onload", "<svg onload=window.__xss_fired=true>"),
    ("obfuscated", "JS bracket notation", "<svg onload=window['__xss_fired']=true>"),
    ("obfuscated", "Hex-escaped name", r"<svg onload=window['\x5f\x5fxss\x5ffired']=true>"),
    ("evasion", "Entity-encoded (gets through)",
     '"><svg onload=&#119;&#105;nd&#111;&#119;.__xss_f&#105;r&#101;d=tr&#117;&#101;>'),
]
GROUP_LABEL = {"benign": "Harmless input", "attack": "Plain attack",
               "obfuscated": "Obfuscated attack", "evasion": "Evasion"}


@app.get("/api/score")
def api_score():
    """What the detector sees for this input, and its verdict at the frozen threshold."""
    q = request.args.get("q", "")
    decoded = decode_once("q=" + quote(q, safe=""))   # == q here; keeps parity with the proxy
    seen = norm(decoded)
    score = float(MODEL.predict_proba([seen])[0, 1])
    return jsonify(score=score, threshold=THRESH, verdict="BLOCK" if score >= THRESH else "ALLOW",
                   model_sees=seen, decoded=decoded)


@app.get("/render")
def render():
    """Fetch the vulnerable app's response and serve it from THIS origin.

    Same-origin means the page can read the iframe's window.__xss_fired, so 'did it
    execute?' is measured live instead of guessed.
    """
    ep = request.args.get("ep", SAFE_ENDPOINT)
    if ep not in ENDPOINTS:
        return Response("bad endpoint", status=400)
    q = request.args.get("q", "")
    url = f"{APP_URL}{ep}?q={quote(q, safe='')}"
    try:
        html = urllib.request.urlopen(url, timeout=3).read().decode("utf-8", "replace")
    except Exception as e:
        return Response(f"<p>upstream error: {e}</p>", mimetype="text/html")
    return Response(html, mimetype="text/html")


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.get("/results")
def results():
    """Every measured table and the figure, read from logs/ (see teach/results.py)."""
    import importlib

    from teach import results as R
    importlib.reload(R)          # pick up newly generated CSVs without a restart
    return Response(R.build(), mimetype="text/html")


@app.get("/figure.png")
def figure():
    from flask import send_file

    from teach.results import FIGURE
    if not FIGURE.exists():
        return Response("figure not generated; run python report/make_figure.py", status=404)
    return send_file(FIGURE, mimetype="image/png")

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XSS Detection Playground</title>
<style>
:root{
  --bg:#0e131f; --card:#161d2c; --sunk:#111725; --edge:#27304a; --edge2:#333f5e;
  --ink:#e8edf7; --mut:#93a1bf; --dim:#6b7793;
  --ok:#35b87a; --bad:#ef5350; --warn:#eab308; --accent:#5b8def;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media(prefers-color-scheme:light){:root{
  --bg:#f5f7fb; --card:#ffffff; --sunk:#f3f5fa; --edge:#e2e7f0; --edge2:#cfd7e6;
  --ink:#141a27; --mut:#5b6882; --dim:#8794ab;
}}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--bg);color:var(--ink);
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:24px 16px 80px}

/* nav */
nav{display:flex;gap:8px;margin-bottom:22px}
nav a{padding:8px 15px;border:1px solid var(--edge);border-radius:999px;text-decoration:none;
  color:var(--mut);font-size:13px;font-weight:600;transition:.15s}
nav a:hover{border-color:var(--edge2);color:var(--ink)}
nav a.on{background:var(--accent);border-color:var(--accent);color:#fff}

/* header */
h1{font-size:25px;line-height:1.2;margin:0 0 6px;letter-spacing:-.4px}
.sub{color:var(--mut);margin:0 0 24px;font-size:14px;max-width:70ch}

/* cards */
.card{background:var(--card);border:1px solid var(--edge);border-radius:14px;padding:18px}
.card + .card{margin-top:14px}
.lbl{font-size:11px;font-weight:700;color:var(--dim);text-transform:uppercase;
  letter-spacing:.08em;margin-bottom:9px}

/* input */
textarea{width:100%;background:var(--sunk);color:var(--ink);border:1px solid var(--edge);
  border-radius:10px;padding:13px 14px;font-family:var(--mono);font-size:13.5px;
  resize:vertical;min-height:62px;line-height:1.5}
textarea:focus{outline:none;border-color:var(--accent)}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:12px}
select{background:var(--sunk);color:var(--ink);border:1px solid var(--edge);border-radius:9px;
  padding:9px 11px;font-size:13.5px;font-family:var(--mono);cursor:pointer}
select:focus{outline:none;border-color:var(--accent)}
button.go{background:var(--accent);border:1px solid var(--accent);color:#fff;border-radius:9px;
  padding:9px 18px;font-size:14px;font-weight:650;cursor:pointer}
button.go:hover{filter:brightness(1.08)}
.hint{color:var(--dim);font-size:12.5px}

/* presets */
.pgroups{display:flex;flex-wrap:wrap;gap:18px;margin-top:16px;
  padding-top:15px;border-top:1px solid var(--edge)}
.pgroup{display:flex;flex-direction:column;gap:7px}
.pgroup .gl{font-size:10.5px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--dim)}
.pgroup .btns{display:flex;gap:6px;flex-wrap:wrap}
.pgroup button{background:var(--sunk);border:1px solid var(--edge);color:var(--ink);
  border-radius:8px;padding:7px 11px;font-size:12.5px;cursor:pointer;transition:.15s}
.pgroup button:hover{border-color:var(--edge2)}
.pgroup button.sel{border-color:var(--accent);color:var(--accent);font-weight:600}
.g-benign .gl{color:var(--ok)} .g-attack .gl{color:var(--warn)}
.g-obfuscated .gl{color:var(--warn)} .g-evasion .gl{color:var(--bad)}

/* detector readout */
.readout{display:grid;grid-template-columns:168px 1fr;gap:22px;align-items:center}
@media(max-width:700px){.readout{grid-template-columns:1fr}}
.scorebox{text-align:center;padding:14px 10px;background:var(--sunk);
  border:1px solid var(--edge);border-radius:12px}
.scorenum{font-size:38px;font-weight:720;letter-spacing:-1.5px;line-height:1;font-variant-numeric:tabular-nums}
.verdict{display:inline-block;margin-top:9px;font-size:12px;font-weight:750;letter-spacing:.07em;
  padding:4px 12px;border-radius:999px}
.verdict.block{background:color-mix(in srgb,var(--bad) 17%,transparent);color:var(--bad)}
.verdict.allow{background:color-mix(in srgb,var(--ok) 17%,transparent);color:var(--ok)}
.meterwrap{position:relative;margin:12px 0 7px}
.meter{height:11px;border-radius:999px;background:var(--sunk);
  border:1px solid var(--edge);overflow:hidden}
.meter>i{display:block;height:100%;border-radius:999px;transition:width .3s ease}
.meter>i.block{background:var(--bad)} .meter>i.allow{background:var(--ok)}
.tick{position:absolute;top:-5px;width:2px;height:21px;background:var(--ink);opacity:.75;border-radius:2px}
.ticklab{position:absolute;top:23px;font-size:10.5px;color:var(--mut);transform:translateX(-50%);white-space:nowrap}
.kv{font-family:var(--mono);font-size:12.5px;color:var(--mut);margin:7px 0;
  word-break:break-all;line-height:1.5}
.kv b{color:var(--ink);font-weight:600}

/* outcome banner */
.banner{margin-top:14px;border-radius:12px;padding:14px 16px;border:1px solid;
  display:flex;gap:12px;align-items:flex-start}
.banner .ico{font-size:17px;line-height:1.3}
.banner .bt{font-weight:700;font-size:14px;margin-bottom:2px}
.banner .bd{font-size:13px;opacity:.92;line-height:1.5}
.b-miss{background:color-mix(in srgb,var(--bad) 11%,transparent);border-color:var(--bad);color:var(--bad)}
.b-catch{background:color-mix(in srgb,var(--ok) 11%,transparent);border-color:var(--ok);color:var(--ok)}
.b-false{background:color-mix(in srgb,var(--warn) 12%,transparent);border-color:var(--warn);color:var(--warn)}
.b-fine{background:var(--sunk);border-color:var(--edge);color:var(--mut)}
.banner .bd{color:var(--ink)}

/* comparison columns */
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px;align-items:start}
.cols .card+.card{margin-top:0}   /* grid gap handles spacing, not the stacking rule */
@media(max-width:820px){.cols{grid-template-columns:1fr}}
.col h3{font-size:14px;margin:0;display:flex;align-items:center;gap:9px;min-height:24px}
.chip{font-size:10.5px;font-weight:750;letter-spacing:.06em;padding:3px 9px;border-radius:999px}
.chip.safe{background:color-mix(in srgb,var(--ok) 17%,transparent);color:var(--ok)}
.chip.unsafe{background:color-mix(in srgb,var(--bad) 17%,transparent);color:var(--bad)}
.col .ep{font-family:var(--mono);font-size:13px}
.frame{width:100%;height:132px;border:1px solid var(--edge);border-radius:10px;
  background:#fff;margin-top:11px;display:block}
.outcome{margin-top:10px;font-size:13px;display:flex;align-items:center;gap:8px}
details{margin-top:10px}
summary{cursor:pointer;font-size:11.5px;color:var(--dim);user-select:none}
summary:hover{color:var(--mut)}
pre.src{margin:8px 0 0;padding:10px;background:var(--sunk);border:1px solid var(--edge);
  border-radius:9px;font-family:var(--mono);font-size:11.5px;color:var(--mut);
  white-space:pre-wrap;word-break:break-all;max-height:130px;overflow:auto;line-height:1.5}
.foot{margin-top:18px;color:var(--mut);font-size:13px;line-height:1.6;
  border-left:3px solid var(--edge2);padding-left:14px}
</style></head><body><div class="wrap">

<nav><a href="/" class="on">Playground</a><a href="/results">Results</a></nav>

<h1>XSS detection playground</h1>
<p class="sub">The detector reads the HTTP request. The browser decides what actually runs.
Send one input to a safe endpoint and an unsafe one and watch those two things disagree.
Everything is local and the only payload is a harmless flag.</p>

<div class="card">
  <div class="lbl">Input sent as <code>?q=</code></div>
  <textarea id="q" spellcheck="false">&lt;img src=x onerror=window.__xss_fired=true&gt;</textarea>
  <div class="controls">
    <span class="hint">Unsafe endpoint:</span>
    <select id="ep"></select>
    <button class="go" id="run">Run</button>
    <span class="hint" id="status"></span>
  </div>
  <div class="pgroups" id="presets"></div>
</div>

<div class="card">
  <div class="lbl">What the detector sees</div>
  <div class="readout">
    <div class="scorebox">
      <div class="scorenum" id="score">0.000</div>
      <div><span class="verdict" id="verdict">&nbsp;</span></div>
    </div>
    <div>
      <div class="meterwrap">
        <div class="meter"><i id="bar" style="width:0%"></i></div>
        <div class="tick" id="tick"></div>
        <div class="ticklab" id="ticklab"></div>
      </div>
      <div style="height:16px"></div>
      <div class="kv">decoded value <b id="decoded"></b></div>
      <div class="kv">model input <b id="seen"></b></div>
    </div>
  </div>
  <div class="banner b-fine" id="banner">
    <div class="ico" id="bico">&#8226;</div>
    <div><div class="bt" id="btitle">Run an input</div>
    <div class="bd" id="bdesc">The verdict above and the two outcomes below are compared here.</div></div>
  </div>
</div>

<div class="cols">
  <div class="card col">
    <h3><span class="chip safe">SAFE</span><span class="ep">/reflect-text</span></h3>
    <iframe class="frame" id="fsafe" sandbox="allow-scripts allow-same-origin"></iframe>
    <div class="outcome" id="rsafe">&nbsp;</div>
    <details><summary>Show what the server returned</summary><pre class="src" id="ssafe"></pre></details>
  </div>
  <div class="card col">
    <h3><span class="chip unsafe">UNSAFE</span><span class="ep" id="unsafeName"></span></h3>
    <iframe class="frame" id="funsafe" sandbox="allow-scripts allow-same-origin"></iframe>
    <div class="outcome" id="runsafe">&nbsp;</div>
    <details><summary>Show what the server returned</summary><pre class="src" id="sunsafe"></pre></details>
  </div>
</div>

<p class="foot">Both columns produced the identical score above, because the request content is
identical. Only the application's use of the value differs. A detector sitting in the network
cannot see that difference, which is the gap this project measures.</p>

<script>
const $=s=>document.querySelector(s);
const EP=%EPS%, PRESETS=%PRESETS%, GROUPS=%GROUPS%;
const esel=$('#ep');
EP.unsafe.forEach(e=>{const o=document.createElement('option');o.value=e;o.textContent=e;esel.appendChild(o)});

const order=['benign','attack','obfuscated','evasion'];
const host=$('#presets');
order.forEach(g=>{
  const items=PRESETS.filter(p=>p[0]===g); if(!items.length) return;
  const d=document.createElement('div'); d.className='pgroup g-'+g;
  d.innerHTML='<div class="gl">'+GROUPS[g]+'</div>';
  const bs=document.createElement('div'); bs.className='btns';
  items.forEach(([,label,val])=>{
    const b=document.createElement('button'); b.textContent=label;
    b.onclick=()=>{document.querySelectorAll('.pgroup button').forEach(x=>x.classList.remove('sel'));
      b.classList.add('sel'); $('#q').value=val; run()};
    bs.appendChild(b)});
  d.appendChild(bs); host.appendChild(d)});

const escHtml=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const markerFired=f=>{try{return f.contentWindow.__xss_fired===true}catch(e){return null}};

let state={verdict:null,safe:null,unsafe:null};

function banner(){
  const {verdict,unsafe}=state;
  if(verdict===null||unsafe===null) return;
  const el=$('#banner');
  el.className='banner ';
  let ico,t,d;
  if(verdict==='ALLOW'&&unsafe){ el.className+='b-miss'; ico='&#9888;';
    t='Missed. The detector allowed an input that executed.';
    d='The score fell below the threshold, so this request would have been forwarded, '+
      'and the browser still ran the marker. This is the robustness gap the experiment measures.'}
  else if(verdict==='BLOCK'&&unsafe){ el.className+='b-catch'; ico='&#10003;';
    t='Caught. Flagged, and it really does execute.';
    d='Score at or above the threshold, and the browser confirms the marker ran on the unsafe endpoint.'}
  else if(verdict==='BLOCK'&&!unsafe){ el.className+='b-false'; ico='&#9888;';
    t='False alarm. Flagged, but nothing executed.';
    d='The detector reacted to how the text looks. On this endpoint it produced no effect in the browser. '+
      'Across the test set the baseline flags 79 to 98 percent of such inert inputs.'}
  else { el.className+='b-fine'; ico='&#8226;';
    t='Allowed, and nothing executed.';
    d='The expected result for harmless input.'}
  $('#bico').innerHTML=ico; $('#btitle').innerHTML=t; $('#bdesc').innerHTML=d;
}

async function run(){
  const q=$('#q').value, unsafe=esel.value;
  $('#unsafeName').textContent=unsafe;
  $('#status').textContent='running...';
  state={verdict:null,safe:null,unsafe:null};

  const d=await (await fetch('/api/score?q='+encodeURIComponent(q))).json();
  $('#decoded').textContent=d.decoded||'(empty)';
  $('#seen').textContent=d.model_sees||'(empty)';
  $('#score').textContent=d.score.toFixed(3);
  const blk=d.verdict==='BLOCK';
  $('#score').style.color='var(--'+(blk?'bad':'ok')+')';
  const v=$('#verdict'); v.textContent=d.verdict; v.className='verdict '+(blk?'block':'allow');
  const bar=$('#bar'); bar.style.width=Math.max(2,Math.round(d.score*100))+'%';
  bar.className=blk?'block':'allow';
  const pct=(d.threshold*100).toFixed(1)+'%';
  $('#tick').style.left=pct; $('#ticklab').style.left=pct;
  $('#ticklab').textContent='threshold '+d.threshold.toFixed(3);
  state.verdict=d.verdict;

  await Promise.all([
    loadFrame($('#fsafe'),'/reflect-text',q,$('#rsafe'),$('#ssafe'),'safe'),
    loadFrame($('#funsafe'),unsafe,q,$('#runsafe'),$('#sunsafe'),'unsafe')]);
  $('#status').textContent='';
  banner();
}

function loadFrame(frame,ep,q,out,src,key){
  return new Promise(async resolve=>{
    out.innerHTML='<span class="hint">rendering...</span>'; src.textContent='';
    const url='/render?ep='+encodeURIComponent(ep)+'&q='+encodeURIComponent(q);
    try{const t=await (await fetch(url)).text(); src.innerHTML=escHtml(t);}catch(e){}
    frame.onload=()=>setTimeout(()=>{
      const fired=markerFired(frame)===true;
      state[key]=fired;
      out.innerHTML=fired
        ? '<span class="chip unsafe">EXECUTED</span><span>the marker ran in the browser</span>'
        : '<span class="chip safe">NO EXECUTION</span><span>shown as text, nothing ran</span>';
      resolve();
    },260);
    frame.src=url;
  });
}

$('#run').onclick=run; esel.onchange=run; esel.selectedIndex=0; run();
</script>
</div></body></html>"""
import json as _json  # noqa: E402

# json.dumps emits valid JS literals, so the page needs no JSON.parse and no hand-escaping.
PAGE = (PAGE
        .replace("%EPS%", _json.dumps({"unsafe": UNSAFE_ENDPOINTS}))
        .replace("%PRESETS%", _json.dumps(PRESETS))
        .replace("%GROUPS%", _json.dumps(GROUP_LABEL)))

if __name__ == "__main__":
    print("playground -> http://127.0.0.1:5001  (needs vuln_app on :5000)")
    app.run(host="127.0.0.1", port=5001, threaded=True)
