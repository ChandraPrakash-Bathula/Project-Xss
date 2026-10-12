"""Cross-context verification against the SECOND app (:5002).

App 1 has four reflected sinks reached the same way. App 2 adds three that app 1 cannot
express, and they are the ones that matter for the "only one application" objection:

  html_body_partial_sanitise  a filter that strips "<script" and nothing else, the most
                              common real half-measure
  json_then_dom_innerhtml     the server returns JSON and the BROWSER writes it with
                              innerHTML, so the sink is client-side and the response the
                              proxy sees contains no payload at all
  stored_html_body_raw        the payload arrives in a POST and is delivered by a later
                              GET, so the request carrying it and the response delivering
                              it are different transactions

That last one is the clearest case of something a request-side detector cannot resolve:
at the moment of the POST there is no response to inspect, and at the moment of the GET
there is no payload in the request.

    python verify/cross_context_app2.py [--limit N] [--resume]
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

APP2 = "http://127.0.0.1:5002"
OUT = LOGS / "cross_context_app2.csv"

TARGETS = [
    ("/s/search", "html_text_escaped"),
    ("/s/profile", "html_body_raw"),
    ("/s/filter", "attribute_double_quoted"),
    ("/s/config", "js_string_literal"),
    ("/s/preview", "html_body_partial_sanitise"),
    ("/s/client", "json_then_dom_innerhtml"),
    ("/s/note", "stored_html_body_raw"),
]
SETTLE_MS = 200
NAV_MS = 5000
COLS = ["payload_id", "norm_text", "group_id", "split", "family", "subtype",
        "endpoint", "sink", "trigger_type", "fired_load", "fired_interact", "note"]


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


def store_note(text):
    """POST the payload, return the GET url that will later serve it."""
    data = urllib.parse.urlencode({"term": text}).encode()
    req = urllib.request.Request(APP2 + "/s/note", data=data)
    with urllib.request.urlopen(req, timeout=5) as r:
        return APP2 + json.load(r)["url"]


def visit(ctx, url, event):
    load = after = False
    note = ""
    page = None
    try:
        page = ctx.new_page()
        page.set_default_timeout(NAV_MS)
        page.on("dialog", lambda d: d.dismiss())
        page.goto(url, wait_until="load", timeout=NAV_MS)
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
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    try:
        urllib.request.urlopen(APP2 + "/s/sinks", timeout=3)
    except Exception:
        sys.exit("app 2 is not running. start it: ./venv/bin/python app/vuln_app2.py &")

    pay = pick(args.train, args.test)
    if args.limit:
        pay = pay.head(args.limit)

    done = set()
    if args.resume and OUT.exists():
        prev = pd.read_csv(OUT, keep_default_na=False)
        done = set(zip(prev.payload_id, prev.endpoint))
    new_file = not (OUT.exists() and args.resume)

    total = len(pay) * len(TARGETS)
    print(f"{len(pay)} payloads x {len(TARGETS)} app-2 sinks = {total} checks"
          + (f"  ({len(done)} done)" if done else ""))

    fh = open(OUT, "w" if new_file else "a", newline="", buffering=1)
    w = csv.writer(fh)
    if new_file:
        w.writerow(COLS)

    n = fl = fi = errs = consec = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context()
        for r in pay.itertuples():
            event = event_of(r.trigger_type)
            for ep, sink in TARGETS:
                if (r.payload_id, ep) in done:
                    continue
                if n and n % 400 == 0:                  # recycle: one wedged browser must
                    try:                                 # not poison the rest of the run
                        ctx.close(); browser.close()
                    except Exception:
                        pass
                    browser = p.chromium.launch(); ctx = browser.new_context()
                n += 1
                try:
                    if ep == "/s/note":
                        url = store_note(r.norm_text)    # POST now, delivered on the GET
                    else:
                        url = f"{APP2}{ep}?term={urllib.parse.quote(r.norm_text, safe='')}"
                    load, after, note = visit(ctx, url, event)
                except Exception as e:
                    load, after, note = False, False, type(e).__name__
                fl += load
                fi += after
                errs += bool(note)
                consec = consec + 1 if note else 0
                if consec >= 25:
                    print(f"  aborting after {consec} consecutive failures ({note})")
                    fh.close()
                    return
                w.writerow([r.payload_id, r.norm_text, r.group_id, r.split, r.family,
                            r.subtype, ep, sink, r.trigger_type, int(load), int(after), note])
                if n % 400 == 0:
                    print(f"  {n}/{total}  load={fl} interact={fi} err={errs}")
        ctx.close()
        browser.close()
    fh.close()
    print(f"\ndone: {n} checks, fired on load {fl}, after interaction {fi}, errors {errs}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
