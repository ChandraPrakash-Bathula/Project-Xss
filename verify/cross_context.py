"""Send every payload to EVERY endpoint, and record execution for each combination.

Why this exists. In manifest.csv each variant carries one intended endpoint, chosen by the
transformation that produced it: js_string_breakout always targets /reflect-js,
entity_in_attr_value always /reflect-attr. Payload and sink are therefore almost
collinear, and any model given both can recover the sink from the payload. Measuring "the
value of context" on that data answers nothing.

Here the payload and the sink are crossed by construction. The same string is sent to all
five contexts and verified in each, so the sink carries information the payload does not.
This is the paired-case experiment generalised from 8 pairs to the whole payload set.

Both labels are recorded in one visit: fired on plain render, and fired after the gated
event is dispatched. Results stream to logs/cross_context.csv and --resume skips what is
already there.

    python verify/cross_context.py [--limit N] [--resume]
"""
import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import DATA, LOGS, MARKER_CHECK, PROXY_URL, SEED  # noqa: E402
from verify.browser_check import _DISPATCH_JS, event_of  # noqa: E402

OUT = LOGS / "cross_context.csv"
ENDPOINTS = ["/reflect-text", "/reflect-html", "/reflect-attr", "/reflect-js", "/reflect-decode"]
SINK = {
    "/reflect-text": "html_text_escaped",
    "/reflect-html": "html_body_raw",
    "/reflect-attr": "attribute_double_quoted",
    "/reflect-js": "js_string_literal",
    "/reflect-decode": "html_body_raw_double_decode",
}
SETTLE_MS = 180
NAV_MS = 5000
COLS = ["payload_id", "norm_text", "wire_q", "group_id", "split", "family", "subtype",
        "endpoint", "sink", "trigger_type", "fired_load", "fired_interact", "note"]


def pick(n_train, n_test):
    """Distinct payloads, keeping each one's own wire form so transport encoding survives."""
    m = pd.read_csv(DATA / "manifest.csv", keep_default_na=False)
    a = m[(m.label_kind == "attack_variant") & (m.wire_q != "")]
    a = a.drop_duplicates("norm_text")
    te = a[a.split == "test"]
    tr = a[a.split.isin(["train", "val"])]
    if n_test:
        te = te.head(n_test)
    if n_train and len(tr) > n_train:
        tr = tr.sample(n=n_train, random_state=SEED)
    out = pd.concat([tr, te], ignore_index=True)
    out = out.rename(columns={"id": "payload_id"})
    return out[["payload_id", "norm_text", "wire_q", "group_id", "split", "family",
                "subtype", "trigger_type"]]


def visit(ctx, url, event):
    """One check. Every call is bounded: a hung page must not wedge a multi-hour run.

    An earlier run stalled for 16 hours when the machine slept and the browser connection
    died while the script was blocked in a call with no timeout.
    """
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
    ap.add_argument("--train", type=int, default=500, help="distinct train/val payloads")
    ap.add_argument("--test", type=int, default=0, help="distinct test payloads, 0 = all")
    ap.add_argument("--limit", type=int, default=0, help="cap total payloads (smoke test)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    pay = pick(args.train, args.test)
    if args.limit:
        pay = pay.head(args.limit)

    done = set()
    if args.resume and OUT.exists():
        prev = pd.read_csv(OUT, keep_default_na=False)
        done = set(zip(prev.payload_id, prev.endpoint))
    new_file = not (OUT.exists() and args.resume)

    total = len(pay) * len(ENDPOINTS)
    print(f"{len(pay)} distinct payloads x {len(ENDPOINTS)} endpoints = {total} checks"
          + (f"  ({len(done)} already done)" if done else ""))

    fh = open(OUT, "w" if new_file else "a", newline="", buffering=1)
    w = csv.writer(fh)
    if new_file:
        w.writerow(COLS)

    n = fired_l = fired_i = errs = 0
    RECYCLE = 400          # restart the browser periodically so a wedged one cannot
    consecutive_errs = 0   # poison the rest of the run, and bail out if it keeps failing

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context()
        for r in pay.itertuples():
            event = event_of(r.trigger_type)
            for ep in ENDPOINTS:
                if (r.payload_id, ep) in done:
                    continue
                if n and n % RECYCLE == 0:
                    try:
                        ctx.close(); browser.close()
                    except Exception:
                        pass
                    browser = p.chromium.launch()
                    ctx = browser.new_context()
                n += 1
                load, after, note = visit(ctx, f"{PROXY_URL}{ep}?q={r.wire_q}", event)
                consecutive_errs = consecutive_errs + 1 if note else 0
                if consecutive_errs >= 25:
                    print(f"  aborting: {consecutive_errs} consecutive failures "
                          f"(last: {note}). Re-run with --resume once the servers are back.")
                    fh.close()
                    return
                fired_l += load
                fired_i += after
                errs += bool(note)
                w.writerow([r.payload_id, r.norm_text, r.wire_q, r.group_id, r.split,
                            r.family, r.subtype, ep, SINK[ep], r.trigger_type,
                            int(load), int(after), note])
                if n % 500 == 0:
                    print(f"  {n}/{total}  fired_load={fired_l}  fired_interact={fired_i}  err={errs}")
        ctx.close()
        browser.close()
    fh.close()
    print(f"\ndone: {n} checks, fired on load {fired_l}, fired after interaction {fired_i}, errors {errs}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
