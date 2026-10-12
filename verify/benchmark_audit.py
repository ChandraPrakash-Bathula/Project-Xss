"""Audit public XSS benchmarks: how many rows labelled "attack" actually do anything?

Every result so far concerns OUR payload set, transformed by US. The sharper question is
about the data the field publishes on. This takes rows verbatim from the public datasets,
with no marker conversion and no transformation, serves each one through every reflected
context of the local app, and records whether a browser does anything observable.

Adjudication is by DIALOG OBSERVATION. These rows call alert/prompt/confirm rather than
our marker, so the harness listens for a dialog event and dismisses it. Nothing we wrote
runs; we only observe whether the browser raised one. 98% of Kaggle attack rows and 66%
of the PayloadBox list contain such a call, so the method can adjudicate most of the data.
Rows without one are recorded as not_observable and excluded from the rate rather than
counted as failures, which keeps the headline conservative.

A row counts as EXECUTING if it produces a dialog in ANY of the five contexts, which is
the most generous possible reading in the dataset's favour.

    python verify/benchmark_audit.py [--n-kaggle 300] [--n-payloadbox 200] [--limit N]
                                     [--resume]
    -> logs/benchmark_audit.csv
"""
import argparse
import csv
import re
import sys
import urllib.parse
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import DATA, LOGS, PROXY_URL, SEED  # noqa: E402
from verify.browser_check import _DISPATCH_JS  # noqa: E402

OUT = LOGS / "benchmark_audit.csv"
SINKS = [("/reflect-text", "html_text_escaped"),
         ("/reflect-html", "html_body_raw"),
         ("/reflect-attr", "attribute_double_quoted"),
         ("/reflect-js", "js_string_literal"),
         ("/reflect-decode", "html_body_raw_double_decode")]
DIALOG = re.compile(r"\b(?:alert|prompt|confirm)\s*\(", re.I)
HANDLER = re.compile(r"\bon([a-z]+)\s*=", re.I)
AUTO = {"load", "error", "toggle", "pageshow", "animationstart", "animationend",
        "transitionend", "loadstart", "canplay", "playing", "loadeddata"}
SETTLE_MS, NAV_MS = 220, 5000
COLS = ["row_id", "source", "text", "observable", "sink", "fired_load", "fired_interact", "note"]


def event_of(text):
    m = HANDLER.search(text)
    if not m:
        return None
    h = m.group(1).lower()
    if h in AUTO:
        return None
    if h == "focus" and "autofocus" in text.lower():
        return None
    return h


def sample(n_kaggle, n_payloadbox):
    rows = []
    k = pd.read_csv(DATA / "raw" / "kaggle" / "XSS_dataset.csv")
    k = k[k.Label == 1].copy()
    k["text"] = k.Sentence.astype(str)
    k = k[k.text.str.len().between(3, 500)]
    k = k.drop_duplicates("text").sample(n=min(n_kaggle, len(k)), random_state=SEED)
    for r in k.itertuples():
        rows.append(("kaggle", str(r._1), r.text))

    pb = [l for l in (DATA / "raw" / "seclists" / "XSS-payloadbox.txt").read_text(
        encoding="utf-8", errors="replace").splitlines() if 3 <= len(l.strip()) <= 500]
    pb = pd.Series(sorted(set(pb))).sample(n=min(n_payloadbox, len(set(pb))),
                                           random_state=SEED)
    for i, t in pb.items():
        rows.append(("payloadbox", f"line{i}", t))
    return rows


def visit(ctx, url, event):
    """Navigate and observe. Returns (dialog_on_load, dialog_after_event, note)."""
    seen = {"n": 0}
    load = after = False
    note = ""
    page = None
    try:
        page = ctx.new_page()
        page.set_default_timeout(NAV_MS)
        page.on("dialog", lambda d: (seen.__setitem__("n", seen["n"] + 1), d.dismiss()))
        page.goto(url, wait_until="load", timeout=NAV_MS)
        page.wait_for_timeout(SETTLE_MS)
        load = seen["n"] > 0
        if not load and event:
            page.evaluate(_DISPATCH_JS, event)
            page.wait_for_timeout(SETTLE_MS)
            after = seen["n"] > 0
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
    ap.add_argument("--n-kaggle", type=int, default=300)
    ap.add_argument("--n-payloadbox", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    rows = sample(args.n_kaggle, args.n_payloadbox)
    if args.limit:
        rows = rows[:args.limit]

    done = set()
    if args.resume and OUT.exists():
        prev = pd.read_csv(OUT, keep_default_na=False)
        done = set(zip(prev.row_id.astype(str), prev.sink))
    new = not (OUT.exists() and args.resume)

    print(f"{len(rows)} public rows x {len(SINKS)} contexts = {len(rows) * len(SINKS)} checks")
    fh = open(OUT, "w" if new else "a", newline="", buffering=1)
    w = csv.writer(fh)
    if new:
        w.writerow(COLS)

    n = errs = consec = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context()
        for src, rid, text in rows:
            obs = bool(DIALOG.search(text))
            ev = event_of(text)
            for ep, sink in SINKS:
                key = (f"{src}:{rid}", sink)
                if key in done:
                    continue
                if n and n % 400 == 0:
                    try:
                        ctx.close(); browser.close()
                    except Exception:
                        pass
                    browser = p.chromium.launch(); ctx = browser.new_context()
                n += 1
                url = f"{PROXY_URL}{ep}?q={urllib.parse.quote(text, safe='')}"
                load, after, note = visit(ctx, url, ev)
                errs += bool(note)
                consec = consec + 1 if note else 0
                if consec >= 25:
                    print(f"  aborting after {consec} consecutive failures ({note})")
                    fh.close(); return
                w.writerow([f"{src}:{rid}", src, text, int(obs), sink,
                            int(load), int(after), note])
                if n % 400 == 0:
                    print(f"  {n}/{len(rows) * len(SINKS)}  errors={errs}", flush=True)
        ctx.close(); browser.close()
    fh.close()
    print(f"\ndone: {n} checks, {errs} errors -> {OUT}")


if __name__ == "__main__":
    main()
