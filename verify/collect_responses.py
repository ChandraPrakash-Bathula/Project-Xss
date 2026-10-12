"""Capture the HTTP RESPONSE BODY for every payload x sink pair.

Our central claim is that a request-side detector cannot see the output context. The
obvious follow-up from a reviewer is: then inspect responses instead. That is testable,
and it is the constructive half of the paper, so it should be measured rather than
argued.

For a reflected sink the response body contains the payload already placed in its
rendering context: escaped as text, raw inside a tag, inside an attribute, inside a
script. If execution is far more predictable from the response than from the request,
the recommendation is concrete. If it is not, that is equally worth reporting.

No browser needed: this is plain HTTP. Rows mirror logs/cross_context.csv so the two
join on (payload_id, sink).

    python verify/collect_responses.py [--limit N] [--resume]
    -> logs/responses.csv   (payload_id, sink, status, resp_body)
"""
import argparse
import csv
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import APP_URL, LOGS  # noqa: E402

OUT = LOGS / "responses.csv"
MAX_BODY = 4000          # bodies are small; cap so one pathological page cannot dominate


def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read().decode("utf-8", "replace")[:MAX_BODY], ""
    except urllib.error.HTTPError as e:
        return e.code, "", "HTTPError"
    except Exception as e:
        return 0, "", type(e).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    c = pd.read_csv(LOGS / "cross_context.csv", keep_default_na=False)
    c = c[c.note == ""]
    if args.limit:
        c = c.head(args.limit)

    done = set()
    if args.resume and OUT.exists():
        prev = pd.read_csv(OUT, keep_default_na=False)
        done = set(zip(prev.payload_id.astype(str), prev.sink))
    new = not (OUT.exists() and args.resume)

    print(f"{len(c)} payload x sink pairs to fetch"
          + (f" ({len(done)} already done)" if done else ""))
    fh = open(OUT, "w" if new else "a", newline="", buffering=1)
    w = csv.writer(fh)
    if new:
        w.writerow(["payload_id", "sink", "endpoint", "status", "note", "resp_body"])

    n = errs = 0
    for r in c.itertuples():
        if (str(r.payload_id), r.sink) in done:
            continue
        n += 1
        # the app is addressed directly: we want what the server emits, not proxy timing
        status, body, note = fetch(f"{APP_URL}{r.endpoint}?q={r.wire_q}")
        errs += bool(note)
        w.writerow([r.payload_id, r.sink, r.endpoint, status, note, body])
        if n % 1000 == 0:
            print(f"  {n} fetched, {errs} errors", flush=True)
    fh.close()
    print(f"\ndone: {n} fetched, {errs} errors -> {OUT}")


if __name__ == "__main__":
    main()
