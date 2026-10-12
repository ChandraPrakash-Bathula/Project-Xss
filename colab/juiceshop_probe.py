"""Colab: run the context experiment against OWASP Juice Shop, a third-party application.

Paste into ONE Colab cell and run. No GPU needed; this needs Colab for MEMORY, since
Juice Shop wants ~500 MB plus Chromium and will not fit on an 8 GB laptop in swap.

WHY
Both of our test apps were written by us, so we chose which sinks escape and which do
not. The template-engine harness externalised the escaping decision to ejs/handlebars/
mustache/pug, but we still wrote the page around it. Juice Shop closes the gap: an OWASP
project, deliberately vulnerable by design, written by people with no stake in our
argument, and its XSS is DOM-based through an Angular SPA, a sink family our apps barely
cover.

SELF-VALIDATION (important)
This script cannot be tested on the machine that wrote it, so it validates itself before
trusting any result:

  1. a POSITIVE CONTROL payload, the documented Juice Shop DOM-XSS vector rewritten to
     set our inert marker, is fired first. If it does NOT execute, the harness or the
     route is wrong and the script ABORTS rather than reporting a misleading 0%.
  2. several candidate routes are probed and only those where the control fires are used.
  3. a NEGATIVE CONTROL route with no injection point must stay at 0.

Upload data/manifest.csv when asked. Output: juiceshop_context.csv.
Everything stays inside the Colab VM and the only payload is window.__xss_fired = true.
"""
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

subprocess.run([sys.executable, "-m", "pip", "-q", "install", "playwright", "pandas"],
               check=False)
subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False)
subprocess.run([sys.executable, "-m", "playwright", "install-deps", "chromium"], check=False)

import pandas as pd                                        # noqa: E402
from playwright.sync_api import sync_playwright            # noqa: E402

JS_PORT = 3017
APP = f"http://127.0.0.1:{JS_PORT}"
JS_ROOT = []
LAUNCH = {}
RESTARTS = [0]


def spawn():
    return subprocess.Popen(LAUNCH["cmd"], shell=True, cwd=LAUNCH["cwd"],
                            stdout=LAUNCH["log"], stderr=LAUNCH["log"],
                            env=dict(os.environ, NODE_ENV="unsafe", PORT=str(JS_PORT)))


def watchdog(holder, stop):
    """Juice Shop can throw an unhandled error in one of its challenge-verification
    routes and exit. Without this a single crash silently turns every later check into
    a connection error. The restarts are counted and reported, never hidden."""
    import threading
    while not stop.is_set():
        if holder["p"].poll() is not None:
            RESTARTS[0] += 1
            holder["p"] = spawn()
            for _ in range(60):
                if alive():
                    break
                time.sleep(1)
        stop.wait(1.0)            # filled in by start_juice_shop so failures can show its log


def alive():
    try:
        urllib.request.urlopen(APP + "/rest/admin/application-version", timeout=3)
        return True
    except Exception:
        return False
SEED, N_PAYLOADS = 42, 250
SETTLE_MS, NAV_MS = 1200, 20000      # Angular SPA: much slower to settle than our Flask apps

# The documented Juice Shop DOM-XSS vector, with alert() swapped for our inert marker.
# An iframe has its own window object, so the flag must be set on the PARENT or the
# page-level check never sees it. (The first version set it inside the iframe and read
# it from the top window, which can only ever report "inert".)
CONTROL = '<iframe src="javascript:parent.window.__xss_fired=true">'
# Routes to probe. Only those where CONTROL fires are used for the real sweep.
CANDIDATES = [
    ("/#/search?q={q}",        "juiceshop_search_dom"),
    ("/#/track-result?id={q}", "juiceshop_track_dom"),
    ("/#/contact",             "juiceshop_contact_control"),   # negative control
]


def sh(cmd, **kw):
    """Run a shell command and ALWAYS surface its output. The previous version hid the
    npm error behind a bare 'install failed', which made diagnosis impossible."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(f"$ {cmd}\n--- stdout ---\n{r.stdout[-2500:]}\n--- stderr ---\n{r.stderr[-2500:]}")
    return r


# Juice Shop 20.x requires Node 22-26, and Colab ships 20. Must be >= 22.12.0: Juice
# Shop's @otplib dependency does require() of an ES module, which only works unflagged
# from 22.12 onward. 22.11.0 installs fine and then dies with ERR_REQUIRE_ESM.
NODE_FALLBACK = "v22.23.3"
NODE_DIR = f"/opt/node-{NODE_FALLBACK}-linux-x64"


def node_major(node_bin="node"):
    r = subprocess.run(f"{node_bin} --version", shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return int(r.stdout.strip().lstrip("v").split(".")[0])
    except Exception:
        return None


def ensure_supported_node():
    """Return a path to a node binary in Juice Shop's supported range (22-26).

    Colab ships Node 20.19, and Juice Shop refuses to boot outside 22-26 with
    'Exiting due to unsatisfied precondition'. Rather than apt (slow, needs the
    NodeSource repo) we drop an official binary tarball into /opt and use it directly.
    """
    nm = node_major()
    print(f"system node major: {nm}")
    nv = subprocess.run("node --version", shell=True, capture_output=True, text=True).stdout.strip()
    ok_minor = True
    if nm == 22:
        try:
            ok_minor = int(nv.lstrip("v").split(".")[1]) >= 12   # require(esm) support
        except Exception:
            ok_minor = False
    if nm is not None and 22 <= nm <= 26 and ok_minor:
        return "node", nm
    cand = os.path.join(NODE_DIR, "bin", "node")
    if not os.path.exists(cand):
        url = (f"https://nodejs.org/dist/{NODE_FALLBACK}/"
               f"node-{NODE_FALLBACK}-linux-x64.tar.xz")
        print(f"installing {NODE_FALLBACK} (Colab's {nm} is outside 22-26) ...", flush=True)
        r = sh(f"curl -fsSL -o /tmp/node.tar.xz '{url}' && mkdir -p /opt && "
               f"tar xJf /tmp/node.tar.xz -C /opt")
        if r.returncode != 0 or not os.path.exists(cand):
            print("could not install a supported Node")
            return None, nm
    got = node_major(cand)
    print(f"using {cand} (major {got})")
    return cand, got


def start_juice_shop():
    """Install from a PREBUILT release tarball and run it on a SUPPORTED Node.

    Two things bite here and both are handled explicitly:
      * `npm install juice-shop` builds from source and fails on Colab, so we take the
        project's prebuilt linux-x64 bundle instead.
      * Juice Shop 20.x refuses to boot on Node outside 22-26, and Colab ships 20.19.
        ensure_supported_node() drops an official Node 22 binary into /opt and we run
        the app with that, leaving Colab's own node untouched.
    """
    JS_VER = "20.2.0"
    node_bin, nm = ensure_supported_node()
    if node_bin is None:
        print("ABORT: no Node in the supported range 22-26 could be provisioned.")
        return None
    pick = min([22, 24, 26], key=lambda v: (abs(v - nm), v))
    root = f"juice-shop_{JS_VER}_node{pick}"

    if not os.path.exists(os.path.join(root, "build", "app.js")):
        sh(f"rm -rf {root} js_extract js.tgz")
        asset = f"juice-shop-{JS_VER}_node{pick}_linux_x64.tgz"
        url = f"https://github.com/juice-shop/juice-shop/releases/download/v{JS_VER}/{asset}"
        print(f"downloading prebuilt {asset} (~95 MB) ...", flush=True)
        r = sh(f"curl -fsSL -o js.tgz '{url}'")
        if r.returncode != 0 or not os.path.exists("js.tgz"):
            print("ABORT: download failed; see output above")
            return None
        sh("mkdir -p js_extract && tar xzf js.tgz -C js_extract")
        inner = [d for d in os.listdir("js_extract")
                 if os.path.isdir(os.path.join("js_extract", d))]
        if not inner:
            print("ABORT: tarball contained no directory")
            return None
        os.rename(os.path.join("js_extract", inner[0]), root)
        print(f"extracted to {root}/")

    sh("pkill -f 'build/app.js' || true")
    sh(f"fuser -k {JS_PORT}/tcp || true")
    time.sleep(2)
    if alive():
        print(f"ABORT: something is still answering on :{JS_PORT} before we started ours.")
        return None
    JS_ROOT.clear(); JS_ROOT.append(root)
    print(f"starting juice-shop with {node_bin} ...", flush=True)
    log = open("juice.log", "w")
    LAUNCH.clear()
    LAUNCH.update(cmd=f"{node_bin} build/app.js", cwd=root, log=log)
    p = spawn()
    for i in range(300):
        try:
            urllib.request.urlopen(APP + "/rest/admin/application-version", timeout=2)
            print(f"juice-shop up after {i}s")
            return p
        except Exception:
            if p.poll() is not None:
                tail = open("juice.log").read()[-2500:]
                print("ABORT: juice-shop exited during startup. Last log lines:\n" + tail)
                return None
            time.sleep(1)
    print("ABORT: juice-shop never became reachable. Log tail:\n"
          + open("juice.log").read()[-2500:])
    return None


# ---------------------------------------------------------------- browser worker
# Colab notebooks run inside an asyncio event loop, and Playwright's sync API refuses
# to start there ("use the Async API instead"). Rather than rewrite everything async,
# the browser work runs in a plain subprocess, which has no running loop.
WORKER = r"""
import csv, json, sys, urllib.parse
import pandas as pd
from playwright.sync_api import sync_playwright

APP, CONTROL, MANIFEST, OUT, N, SEED = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
CANDIDATES = json.loads(sys.argv[7])
SETTLE_MS, NAV_MS = 1200, 20000

def check(ctx, url, diag=False):
    # diag also reports HTTP status, rendered body size and title, so "the route
    # never rendered" can be told apart from "it rendered and stayed safe".
    page = None
    try:
        page = ctx.new_page()
        page.set_default_timeout(NAV_MS)
        page.on("dialog", lambda d: d.dismiss())
        resp = page.goto(url, wait_until="networkidle", timeout=NAV_MS)
        page.wait_for_timeout(SETTLE_MS)
        fired = bool(page.evaluate("window.__xss_fired === true"))
        if diag:
            st = resp.status if resp is not None else "?"
            blen = page.evaluate("document.body ? document.body.innerHTML.length : -1")
            return fired, f"http={st} body_len={blen} title={(page.title() or '')[:40]!r}"
        return fired, ""
    except Exception as e:
        return False, type(e).__name__
    finally:
        if page is not None:
            try: page.close()
            except Exception: pass

with sync_playwright() as pw:
    browser = pw.chromium.launch(args=["--no-sandbox"])
    ctx = browser.new_context()

    print("POSITIVE CONTROL: does the documented DOM-XSS vector fire?", flush=True)
    enc = urllib.parse.quote(CONTROL, safe="")
    live = []
    for tpl, sink in CANDIDATES:
        fired, note = check(ctx, APP + tpl.replace("{q}", enc), diag=True)
        print(f"   {'FIRED ' if fired else 'inert '} {sink:<28} {note}", flush=True)
        if fired and "control" not in sink: live.append([tpl, sink])
        if fired and "control" in sink: print("   WARNING: negative control fired; harness suspect", flush=True)

    if not live:
        try:
            urllib.request.urlopen(APP + "/rest/admin/application-version", timeout=5)
            print("ABORT_NO_CONTROL (server still alive: routes are wrong)", flush=True)
        except Exception:
            print("ABORT_SERVER_DIED_MIDWAY (server went away during the control phase)", flush=True)
        browser.close(); sys.exit(0)
    print(f"using {len(live)} validated sink(s): {[s for _, s in live]}", flush=True)

    m = pd.read_csv(MANIFEST, keep_default_na=False)
    a = m[m.label_kind == "attack_variant"].drop_duplicates("norm_text")
    a = a.sample(n=min(N, len(a)), random_state=SEED)
    total = len(a) * len(live)
    print(f"{len(a)} payloads x {len(live)} sinks = {total} checks", flush=True)

    fh = open(OUT, "w", newline="", buffering=1); w = csv.writer(fh)
    w.writerow(["payload_id","norm_text","group_id","split","family","subtype","sink","fired","note"])
    n = 0
    for r in a.itertuples():
        for tpl, sink in live:
            n += 1
            if n % 200 == 0:
                try: ctx.close(); browser.close()
                except Exception: pass
                browser = pw.chromium.launch(args=["--no-sandbox"]); ctx = browser.new_context()
                print(f"   {n}/{total}", flush=True)
            url = APP + tpl.replace("{q}", urllib.parse.quote(r.norm_text, safe=""))
            fired, note = check(ctx, url)
            w.writerow([r.id, r.norm_text, r.group_id, r.split, r.family, r.subtype, sink, int(fired), note])
    fh.close()
    try: ctx.close(); browser.close()
    except Exception: pass
print("WORKER_DONE", flush=True)
"""


def check(ctx, url):
    """Load one URL, return (marker_fired, note)."""
    page = None
    try:
        page = ctx.new_page()
        page.set_default_timeout(NAV_MS)
        page.on("dialog", lambda d: d.dismiss())
        page.goto(url, wait_until="domcontentloaded", timeout=NAV_MS)
        page.wait_for_timeout(SETTLE_MS)
        return bool(page.evaluate("window.__xss_fired === true")), ""
    except Exception as e:
        return False, type(e).__name__
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:
                pass


def main():
    proc = start_juice_shop()
    if proc is None:
        return
    try:
        if not os.path.exists("manifest.csv"):
            from google.colab import files
            print("upload data/manifest.csv")
            up = files.upload()
            for k in up:
                if "manifest" in k:
                    os.rename(k, "manifest.csv")

        with open("_js_worker.py", "w") as f:
            f.write(WORKER)
        out = "juiceshop_context.csv"
        import threading
        holder, stop = {"p": proc}, threading.Event()
        threading.Thread(target=watchdog, args=(holder, stop), daemon=True).start()
        print(f"juice-shop alive before worker: {alive()}")
        r = subprocess.run([sys.executable, "_js_worker.py", APP, CONTROL, "manifest.csv",
                            out, str(N_PAYLOADS), str(SEED), json.dumps(CANDIDATES)],
                           text=True, capture_output=True)
        print("--- worker output ---")
        print(r.stdout or "(no stdout)")
        if r.stderr.strip():
            print("--- worker stderr ---")
            print(r.stderr[-2500:])
        stop.set()
        print(f"juice-shop alive after worker: {alive()}  | server restarts during run: {RESTARTS[0]}")
        if not alive():
            print("--- juice.log tail ---")
            print(open(os.path.join(JS_ROOT[0], "juice.log")).read()[-2000:]
                  if JS_ROOT and os.path.exists(os.path.join(JS_ROOT[0], "juice.log"))
                  else open("juice.log").read()[-2000:])
        if r.returncode != 0:
            print(f"ABORT: browser worker exited with code {r.returncode}")
            return
        if not os.path.exists(out):
            print("\nABORT: the control payload did not execute on any candidate route.")
            print("That means the routes or the Angular version changed, NOT that Juice")
            print("Shop is safe. Report this rather than a 0% result.")
            print("Juice Shop version:",
                  urllib.request.urlopen(APP + "/rest/admin/application-version",
                                         timeout=5).read().decode()[:120])
            return

        d = pd.read_csv(out, keep_default_na=False)
        print("\nexecution rate by juice-shop sink:")
        print(d.groupby("sink").fired.agg(fired="sum", n="size")
              .assign(rate=lambda x: (x.fired / x.n).round(3)).to_string())
        print(f"errors: {(d.note != '').sum()}")
        print("\nFOR COMPARISON, our measured rates:")
        print("  our app1      escaped 0.000 | attr 0.111 | raw 0.440 | double-decode 0.532")
        print("  our app2      escaped 0.000 | DOM sink 0.447 | stored 0.463")
        print("  3rd-party     all escaped 0.000 | all raw ~0.445")
        try:
            from google.colab import files
            files.download(out)
        except Exception:
            pass
    finally:
        sh("pkill -f 'build/app.js' || true")


if __name__ == "__main__":
    main()
