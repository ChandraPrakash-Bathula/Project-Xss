"""Cross-context sweep against third-party template-engine sinks (:5020).

The escaping behaviour of every sink here is defined by a widely used library (ejs,
handlebars, mustache, pug), not by us. If the context effect measured on our own two
Flask apps is an artefact of how we wrote them, it should weaken or vanish here.

    python verify/cross_context_engines.py [--limit N] [--resume]
    -> logs/cross_context_engines.csv
"""
import argparse
import csv
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import DATA, LOGS, MARKER_CHECK, SEED  # noqa: E402
from verify.browser_check import _DISPATCH_JS, event_of  # noqa: E402

APP = "http://127.0.0.1:5020"
OUT = LOGS / "cross_context_engines.csv"
SETTLE_MS, NAV_MS = 200, 5000
COLS = ["payload_id", "norm_text", "group_id", "split", "family", "subtype",
        "endpoint", "sink", "trigger_type", "fired_load", "fired_interact", "note"]


def sinks():
    with urllib.request.urlopen(APP + "/sinks", timeout=5) as r:
        return json.load(r)


def pick(n_train, n_test):
    m = pd.read_csv(DATA / "manifest.csv", keep_default_na=False)
    a = m[m.label_kind == "attack_variant"].drop_duplicates("norm_text")
    te = a[a.split == "test"]
    tr = a[a.split.isin(["train", "val"])]
    if n_test:
        te = te.head(n_test)
    if n_train and len(tr) > n_train:
        tr = tr.sample(n=n_train, random_state=SEED)
    out = pd.concat([tr, te], ignore_index=True).rename(columns={"id": "payload_id"})
    return out[["payload_id", "norm_text", "group_id", "split", "family", "subtype",
                "trigger_type"]]


def visit(ctx, u, event):
    load = after = False
    note = ""
    page = None
    try:
        page = ctx.new_page()
        page.set_default_timeout(NAV_MS)
        page.on("dialog", lambda d: d.dismiss())
        page.goto(u, wait_until="load", timeout=NAV_MS)
        page.wait_for_timeout(SETTLE_MS)
        load = bool(page.evaluate(MARKER_CHECK))
        if not load and event:
            page.evaluate(_DISPATCH_JS, event)
            page.wait_for_timeout(SETTLE_MS)
            after = bool(page.evaluate(MARKER_CHECK))
    except Exception as e:
        note = type(e).__name__
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:
                pass
    return load, after, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=150)
    ap.add_argument("--test", type=int, default=250)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    try:
        smap = sinks()
    except Exception:
        sys.exit("third-party app not running: cd thirdparty && node engines_app.js &")

    pay = pick(args.train, args.test)
    if args.limit:
        pay = pay.head(args.limit)

    done = set()
    if args.resume and OUT.exists():
        prev = pd.read_csv(OUT, keep_default_na=False)
        done = set(zip(prev.payload_id, prev.endpoint))
    new = not (OUT.exists() and args.resume)

    total = len(pay) * len(smap)
    print(f"{len(pay)} payloads x {len(smap)} third-party sinks = {total} checks")
    fh = open(OUT, "w" if new else "a", newline="", buffering=1)
    w = csv.writer(fh)
    if new:
        w.writerow(COLS)

    n = fl = fi = errs = consec = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context()
        for r in pay.itertuples():
            ev = event_of(r.trigger_type)
            for ep, sink in smap.items():
                if (r.payload_id, ep) in done:
                    continue
                if n and n % 400 == 0:
                    try:
                        ctx.close(); browser.close()
                    except Exception:
                        pass
                    browser = p.chromium.launch(); ctx = browser.new_context()
                n += 1
                u = f"{APP}{ep}?q={urllib.parse.quote(r.norm_text, safe='')}"
                load, after, note = visit(ctx, u, ev)
                fl += load; fi += after; errs += bool(note)
                consec = consec + 1 if note else 0
                if consec >= 25:
                    print(f"  aborting after {consec} consecutive failures ({note})")
                    fh.close(); return
                w.writerow([r.payload_id, r.norm_text, r.group_id, r.split, r.family,
                            r.subtype, ep, sink, r.trigger_type, int(load), int(after), note])
                if n % 400 == 0:
                    print(f"  {n}/{total}  load={fl} interact={fi} err={errs}", flush=True)
        ctx.close(); browser.close()
    fh.close()
    print(f"\ndone: {n} checks, load {fl}, interact {fi}, errors {errs} -> {OUT}")


if __name__ == "__main__":
    main()
