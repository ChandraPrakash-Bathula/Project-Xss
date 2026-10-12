"""Does a synthetic dispatchEvent stand in for a real user action?

Every interaction-gated result rests on an assumption we have not tested: that calling
element.dispatchEvent(new MouseEvent('click')) tells us what a real click would do. A
reviewer is right to challenge this. Synthetic events carry isTrusted=false, they do not
move a real cursor, and they skip hit-testing, so an element that is covered, zero-sized
or off-screen will still receive one.

Here the same payloads are re-tested with Playwright's real input APIs, which drive the
browser's actual input pipeline (locator.click() moves the mouse and presses it,
locator.hover() moves over the element, locator.focus() focuses it). Agreement is
reported as a confusion matrix, and every disagreement is written out for inspection.

If the two agree, the interaction-gated numbers stand and the objection is answered with
a measurement. If they disagree, the headline execution rate is wrong and we report the
real-input figure instead.

    python verify/real_input_check.py [--n 150]   (needs app on APP_PORT, proxy on 8080)
    -> logs/table13_real_input.csv
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import LOGS, MARKER_CHECK, PROXY_URL, SEED  # noqa: E402

NAV_MS, SETTLE_MS = 6000, 250
# events we can genuinely drive through the browser's input pipeline
REAL = {"click": "click", "dblclick": "dblclick",
        "mouseover": "hover", "mouseenter": "hover", "mousemove": "hover",
        "pointerover": "hover", "pointerenter": "hover", "pointermove": "hover",
        "mousedown": "click", "mouseup": "click", "pointerdown": "click",
        "pointerup": "click", "focus": "focus", "focusin": "focus"}


def event_of(t):
    m = re.match(r"on_event:on(\w+)", str(t) or "")
    return m.group(1).lower() if m else None


def synthetic(page, ev):
    """What we have been doing: fabricate and dispatch the event."""
    page.evaluate("""(ev) => {
      for (const el of document.querySelectorAll('[on' + ev + ']')) {
        try {
          if (ev === 'focus') { el.focus(); continue; }
          const M = ['click','dblclick','mousedown','mouseup','mouseover','mouseout',
                     'mousemove','mouseenter','mouseleave','contextmenu'];
          const E = M.includes(ev) ? MouseEvent : (ev.startsWith('key') ? KeyboardEvent : Event);
          el.dispatchEvent(new E(ev, {bubbles: true, cancelable: true}));
        } catch (e) {}
      }
    }""", ev)


def real(page, ev):
    """Drive the browser's actual input pipeline instead."""
    how = REAL.get(ev)
    if how is None:
        return False
    loc = page.locator(f"[on{ev}]")
    n = min(loc.count(), 3)
    acted = False
    for i in range(n):
        el = loc.nth(i)
        try:
            if how == "hover":
                el.hover(timeout=1500, force=True)
            elif how == "focus":
                el.focus(timeout=1500)
            elif how == "dblclick":
                el.dblclick(timeout=1500, force=True, no_wait_after=True)
            else:
                el.click(timeout=1500, force=True, no_wait_after=True)
            acted = True
        except Exception:
            continue
    return acted


def trial(ctx, url, ev, mode):
    page = None
    try:
        page = ctx.new_page()
        page.set_default_timeout(NAV_MS)
        page.on("dialog", lambda d: d.dismiss())
        page.goto(url, wait_until="load", timeout=NAV_MS)
        page.wait_for_timeout(SETTLE_MS)
        if bool(page.evaluate(MARKER_CHECK)):
            return None, "fired_on_load"        # excluded: not interaction-gated
        (synthetic if mode == "synthetic" else real)(page, ev)
        page.wait_for_timeout(SETTLE_MS)
        return bool(page.evaluate(MARKER_CHECK)), ""
    except Exception as e:
        return None, type(e).__name__
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    c = pd.read_csv(LOGS / "cross_context.csv", keep_default_na=False)
    c = c[(c.note == "") & (c.fired_interact == 1) & (c.fired_load == 0)].copy()
    c["ev"] = c.trigger_type.map(event_of)
    c = c[c.ev.isin(REAL)]                       # only events a real user can perform
    c = c.drop_duplicates(["norm_text", "sink"]).sample(
        n=min(args.n, len(c)), random_state=SEED)
    print(f"{len(c)} interaction-gated payloads that fired synthetically, "
          f"events: {sorted(c.ev.unique())}")

    rows = []
    with sync_playwright() as p:
        br = p.chromium.launch()
        ctx = br.new_context()
        for i, r in enumerate(c.itertuples(), 1):
            url = f"{PROXY_URL}{r.endpoint}?q={r.wire_q}"
            s, ns = trial(ctx, url, r.ev, "synthetic")
            t, nt = trial(ctx, url, r.ev, "real")
            rows.append(dict(payload_id=r.payload_id, sink=r.sink, event=r.ev,
                             synthetic=s, real=t, note=(ns or nt),
                             norm_text=r.norm_text[:120]))
            if i % 50 == 0:
                print(f"  {i}/{len(c)}", flush=True)
        ctx.close()
        br.close()

    d = pd.DataFrame(rows)
    d.to_csv(LOGS / "table13_real_input.csv", index=False)
    ok = d[d.synthetic.notna() & d.real.notna()]
    both = int((ok.synthetic & ok.real).sum())
    syn_only = int((ok.synthetic & ~ok.real).sum())
    real_only = int((~ok.synthetic & ok.real).sum())
    neither = int((~ok.synthetic & ~ok.real).sum())

    print(f"\ncomparable trials: {len(ok)} (excluded {len(d) - len(ok)}: fired on load or errored)")
    print("\n                  real: fires   real: no")
    print(f"  synth: fires  {both:>12}{syn_only:>11}")
    print(f"  synth: no     {real_only:>12}{neither:>11}")
    if len(ok):
        agree = (both + neither) / len(ok)
        print(f"\n  agreement {agree:.3f} | real-input execution rate {ok.real.mean():.3f}"
              f" | synthetic {ok.synthetic.mean():.3f}")
        if syn_only:
            print(f"\n  {syn_only} fired ONLY synthetically (synthetic overstates exploitability):")
            for _, r in ok[ok.synthetic & ~ok.real].head(5).iterrows():
                print(f"    [{r.event}] {r.norm_text[:88]}")
    print(f"\n-> {LOGS / 'table13_real_input.csv'}")


if __name__ == "__main__":
    main()
