"""Does "this payload executes" depend on which browser you ask?

Every execution label so far comes from Chromium. Two claims ride on that:

  * that a large share of public payload rows are inert. If those vectors are merely
    Chromium-dead and still work in Firefox or WebKit, the claim is about one engine.
  * that escaping prevents execution. That should hold in every engine, and if it does
    not we need to know before saying so.

Same payloads, same sinks, three engines.

    python verify/cross_browser.py [--n 200]   (needs the app on APP_PORT)
    -> logs/table14_cross_browser.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import LOGS, MARKER_CHECK, PROXY_URL, SEED  # noqa: E402
from verify.browser_check import _DISPATCH_JS, event_of  # noqa: E402

ENGINES = ["chromium", "firefox", "webkit"]
NAV_MS, SETTLE_MS = 8000, 250
OUT = LOGS / "table14_cross_browser.csv"


def visit(ctx, url, event):
    page = None
    try:
        page = ctx.new_page()
        page.set_default_timeout(NAV_MS)
        page.on("dialog", lambda d: d.dismiss())
        page.goto(url, wait_until="load", timeout=NAV_MS)
        page.wait_for_timeout(SETTLE_MS)
        load = bool(page.evaluate(MARKER_CHECK))
        after = False
        if not load and event:
            page.evaluate(_DISPATCH_JS, event)
            page.wait_for_timeout(SETTLE_MS)
            after = bool(page.evaluate(MARKER_CHECK))
        return int(load or after), int(load), ""
    except Exception as e:
        return 0, 0, type(e).__name__
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:
                pass


def preflight():
    """Refuse to run against a dead proxy.

    A previous run silently produced 1000 rows of "0 fired" because :8080 was down.
    A dead endpoint and a safe application are indistinguishable in the output, which is
    the exact failure mode this project is about, so check before collecting anything.
    """
    import urllib.request
    probe = f"{PROXY_URL}/reflect-text?q=ping"
    try:
        body = urllib.request.urlopen(probe, timeout=5).read().decode("utf-8", "replace")
    except Exception as e:
        sys.exit(f"ABORT: proxy not reachable at {PROXY_URL} ({type(e).__name__}). "
                 f"start it:  APP_PORT=5010 ./run.sh start")
    if "ping" not in body:
        sys.exit(f"ABORT: {PROXY_URL} answered but did not reflect; something else is on "
                 f"that port. Body began: {body[:80]!r}")
    print(f"preflight ok: {PROXY_URL} reflects input")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="distinct payloads")
    ap.add_argument("--fresh", action="store_true", help="ignore any existing rows")
    args = ap.parse_args()
    preflight()

    c = pd.read_csv(LOGS / "cross_context.csv", keep_default_na=False)
    c = c[c.note == ""]
    pay = c.drop_duplicates("payload_id").sample(
        n=min(args.n, c.payload_id.nunique()), random_state=SEED).payload_id
    sub = c[c.payload_id.isin(set(pay))]
    print(f"{sub.payload_id.nunique()} payloads x {sub.sink.nunique()} sinks "
          f"x {len(ENGINES)} engines = {len(sub) * len(ENGINES)} checks")

    import csv as _csv
    done = set()
    if OUT.exists() and not args.fresh:
        prev = pd.read_csv(OUT, keep_default_na=False)
        if len(prev) and "engine" in prev:
            done = set(zip(prev.engine, prev.payload_id.astype(str), prev.sink))
            print(f"resuming: {len(done)} checks already on disk")
    new_file = not (OUT.exists() and done)
    fh = open(OUT, "w" if new_file else "a", newline="", buffering=1)
    wr = _csv.writer(fh)
    COLS = ["engine", "payload_id", "sink", "family", "subtype",
            "executes", "fired_load", "note", "norm_text"]
    if new_file:
        wr.writerow(COLS)

    rows = []
    with sync_playwright() as p:
        for eng in ENGINES:
            try:
                br = getattr(p, eng).launch()
            except Exception as e:
                print(f"  {eng}: cannot launch ({type(e).__name__}), skipped")
                continue
            ctx = br.new_context()
            n = fired = 0
            for r in sub.itertuples():
                if (eng, str(r.payload_id), r.sink) in done:
                    continue
                ev = event_of(r.trigger_type)
                ex, on_load, note = visit(ctx, f"{PROXY_URL}{r.endpoint}?q={r.wire_q}", ev)
                rec = dict(engine=eng, payload_id=r.payload_id, sink=r.sink,
                           family=r.family, subtype=r.subtype,
                           executes=ex, fired_load=on_load, note=note,
                           norm_text=r.norm_text[:150])
                rows.append(rec)
                wr.writerow([rec[c] for c in COLS])      # survives an interrupted run
                n += 1
                fired += ex
                if n % 300 == 0:
                    print(f"  {eng}: {n}/{len(sub)} ({fired} fired)", flush=True)
            ctx.close()
            br.close()
            print(f"  {eng}: done, {fired}/{n} executed")

    fh.close()
    d = pd.read_csv(OUT, keep_default_na=False)          # include any resumed rows

    print("\nEXECUTION RATE BY ENGINE AND SINK")
    print(d.pivot_table(index="sink", columns="engine", values="executes",
                        aggfunc="mean").round(3).to_string())

    piv = d.pivot_table(index=["payload_id", "sink"], columns="engine",
                        values="executes", aggfunc="max")
    have = [e for e in ENGINES if e in piv.columns]
    if len(have) > 1:
        agree = piv[have].nunique(axis=1) == 1
        print(f"\nengines agree on {agree.mean():.3f} of {len(piv)} payload-sink pairs"
              f"  ({(~agree).sum()} disagreements)")
        dis = piv[~agree]
        if len(dis):
            print("\nhow often each engine is the ONLY one that executes:")
            for e in have:
                others = [x for x in have if x != e]
                print(f"  {e:<10} {int(((dis[e] == 1) & (dis[others].sum(axis=1) == 0)).sum())}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
