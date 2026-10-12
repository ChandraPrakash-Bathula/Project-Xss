"""Ground-truth labelling in a real browser (Step 5), local and marker-only.

For each ATTACK variant it navigates to  <proxy>/<endpoint>?q=<wire_q>  (proxy in MODE=off,
so the detector never interferes) in a fresh page, then reads window.__xss_fired.

What each column means (kept deliberately separate so Table 1 never conflates them):

  browser_confirmed
      Did the marker fire from plain rendering, with NO synthetic interaction? This is the
      main ground-truth label. For an on_event payload this is expected to be 0.
  interaction_gated
      1 if the payload's only trigger is a user event (onclick, onmouseover, ...). A passive
      request-time detector cannot know whether such an event will ever happen -> a finding.
  browser_confirmed_with_interaction  (only with --interact)
      Second, clearly-separate pass: we synthetically dispatch the gated event, then re-check.
      This measures "would fire IF the user acted", never mixed into browser_confirmed.

FPR-probe (benign) rows are skipped: there is no exploit to confirm (browser_confirmed=NULL).

Requires the Flask app (:5000) and the proxy in MODE=off (:8080) to be running.
Results are streamed to data/browser_results.csv so a crash never loses progress.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, PROXY_URL  # noqa: E402

RESULTS = DATA / "browser_results.csv"
SETTLE_MS = 200          # time for onload/onerror/script to run after navigation
NAV_TIMEOUT_MS = 5000


def event_of(trigger_type):
    """'on_event:onclick' -> 'click'. None for on_load/unknown."""
    m = re.match(r"on_event:on(\w+)", trigger_type or "")
    return m.group(1) if m else None


# Synthetic dispatch for the --interact pass. We fire the event on every element that
# declares the matching handler; focus/blur use the DOM methods.
_DISPATCH_JS = """
(evt) => {
  let n = 0;
  for (const el of document.querySelectorAll('[on' + evt + ']')) {
    n++;
    try {
      if (evt === 'focus') { el.focus(); continue; }
      if (evt === 'blur')  { el.focus(); el.blur(); continue; }
      const Mouse = ['click','dblclick','mousedown','mouseup','mouseover','mouseout',
                     'mousemove','mouseenter','mouseleave','contextmenu'];
      const Ev = Mouse.includes(evt) ? MouseEvent : (evt.startsWith('key') ? KeyboardEvent : Event);
      el.dispatchEvent(new Ev(evt, {bubbles: true, cancelable: true}));
    } catch (e) {}
  }
  return n;
}
"""


def check_one(context, url, event=None, interact=False):
    """Navigate fresh, read the marker. Returns (fired_on_load, fired_after_event, note)."""
    page = context.new_page()
    page.on("dialog", lambda d: d.dismiss())   # a stray alert must never hang the run
    fired_load, fired_after, note = False, False, ""
    try:
        page.goto(url, wait_until="load", timeout=NAV_TIMEOUT_MS)
        page.wait_for_timeout(SETTLE_MS)
        fired_load = bool(page.evaluate("window.__xss_fired === true"))
        if interact and event and not fired_load:
            page.evaluate(_DISPATCH_JS, event)
            page.wait_for_timeout(SETTLE_MS)
            fired_after = bool(page.evaluate("window.__xss_fired === true"))
    except Exception as e:
        note = f"{type(e).__name__}: {str(e)[:80]}"
    finally:
        page.close()
    return fired_load, fired_after, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interact", action="store_true",
                    help="second pass: dispatch gated events, record browser_confirmed_with_interaction")
    ap.add_argument("--limit", type=int, default=0, help="test only the first N variants (smoke test)")
    ap.add_argument("--resume", action="store_true", help="skip ids already in browser_results.csv")
    args = ap.parse_args()

    global RESULTS
    v = pd.read_csv(DATA / "variants_raw.csv", keep_default_na=False)
    todo = v[v.label_kind == "attack_variant"].copy()
    if args.interact:
        # second pass: only interaction-gated variants, written to a SEPARATE file so the
        # main browser_confirmed label is never touched
        todo = todo[todo.trigger_type.str.startswith("on_event")]
        RESULTS = DATA / "browser_results_interact.csv"
    if args.limit:
        todo = todo.head(args.limit)

    done = set()
    if args.resume and RESULTS.exists():
        done = set(pd.read_csv(RESULTS, keep_default_na=False).id)
        todo = todo[~todo.id.isin(done)]
    header_needed = not (RESULTS.exists() and args.resume)

    print(f"to verify: {len(todo)} attack variants"
          + (f" (resuming, {len(done)} already done)" if args.resume else "")
          + (f" (--limit {args.limit})" if args.limit else ""))

    cols = ["id", "endpoint", "subtype", "trigger_type", "browser_confirmed",
            "interaction_gated", "browser_confirmed_with_interaction", "note"]
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        out = open(RESULTS, "a", buffering=1)
        if header_needed:
            out.write(",".join(cols) + "\n")

        n_fired = n_gated = n_interact = n_err = 0
        for i, (_, r) in enumerate(todo.iterrows(), 1):
            url = f"{PROXY_URL}{r.endpoint}?q={r.wire_q}"
            event = event_of(r.trigger_type)
            gated = 1 if event else 0
            fired_load, fired_after, note = check_one(context, url, event, args.interact)

            confirmed = int(fired_load)
            with_interaction = ("NULL" if not args.interact or not gated
                                else int(fired_after or fired_load))
            n_fired += confirmed
            n_gated += gated
            n_interact += 1 if (args.interact and gated and fired_after) else 0
            n_err += 1 if note else 0

            row = [r.id, r.endpoint, r.subtype, r.trigger_type, confirmed, gated,
                   with_interaction, note.replace(",", ";")]
            out.write(",".join(map(str, row)) + "\n")
            if i % 200 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  fired_on_load={n_fired}  gated={n_gated}"
                      + (f"  fired_with_interaction={n_interact}" if args.interact else "")
                      + f"  errors={n_err}")
        out.close()
        context.close()
        browser.close()

    print(f"\ndone -> {RESULTS}")
    print(f"browser_confirmed (fired on load): {n_fired}/{len(todo)} ({n_fired/max(len(todo),1):.0%})")
    print(f"interaction_gated:                 {n_gated}/{len(todo)} ({n_gated/max(len(todo),1):.0%})")
    if args.interact:
        print(f"fired only after interaction:      {n_interact}")


if __name__ == "__main__":
    main()
