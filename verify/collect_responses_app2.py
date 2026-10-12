"""Capture response bodies from app 2, including the stored and DOM paths.

For reflected sinks the response is simply the reply to the request that carried the
payload. The other two classes are the interesting ones and need different handling:

  stored   the payload arrives in a POST and is served by a LATER GET. The delivering
           request contains no payload at all, so we POST first and then capture the
           body of the GET that serves it. If response-side inspection works here, it
           covers exactly the case where request-side inspection has nothing to look at.

  DOM      the server returns a page with no payload in it; client script fetches JSON
           separately and writes it into the document. We capture BOTH the page body and
           the JSON reply, because a proxy would see both on the wire, and report them
           separately so the difference is visible.

    python verify/collect_responses_app2.py [--limit N]
    -> logs/responses_app2.csv
"""
import argparse
import csv
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import LOGS  # noqa: E402

APP2 = "http://127.0.0.1:5002"
OUT = LOGS / "responses_app2.csv"
MAX_BODY = 4000

# sink -> endpoint template. "{q}" is the url-encoded payload.
TEMPLATES = {
    "html_text_escaped": "/s/search?term={q}",
    "html_body_raw": "/s/profile?term={q}",
    "attribute_double_quoted": "/s/filter?term={q}",
    "js_string_literal": "/s/config?term={q}",
    "html_body_partial_sanitise": "/s/preview?term={q}",
    "json_then_dom_innerhtml": "/s/client?term={q}",      # page body, payload NOT in it
    "stored_html_body_raw": None,                          # POST then GET, handled below
}


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.read().decode("utf-8", "replace")[:MAX_BODY], ""
    except Exception as e:
        return "", type(e).__name__


def stored_body(text):
    """POST the payload, then capture the body of the GET that later serves it."""
    data = urllib.parse.urlencode({"term": text}).encode()
    try:
        req = urllib.request.Request(APP2 + "/s/note", data=data)
        nid = json.load(urllib.request.urlopen(req, timeout=5))["id"]
    except Exception as e:
        return "", "", type(e).__name__
    body, note = get(f"{APP2}/s/note/{nid}")
    return body, f"/s/note/{nid}", note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    try:
        urllib.request.urlopen(APP2 + "/s/sinks", timeout=3)
    except Exception:
        sys.exit("app 2 not running: ./venv/bin/python app/vuln_app2.py &")

    c = pd.read_csv(LOGS / "cross_context_app2.csv", keep_default_na=False)
    c = c[c.note == ""]
    if args.limit:
        c = c.head(args.limit)
    print(f"{len(c)} payload x sink pairs")

    fh = open(OUT, "w", newline="", buffering=1)
    w = csv.writer(fh)
    w.writerow(["payload_id", "sink", "delivering_path", "resp_body", "aux_body", "note"])
    n = errs = 0
    for r in c.itertuples():
        enc = urllib.parse.quote(r.norm_text, safe="")
        aux = ""
        if r.sink == "stored_html_body_raw":
            body, path, note = stored_body(r.norm_text)
        else:
            tpl = TEMPLATES.get(r.sink)
            if tpl is None:
                continue
            path = tpl.replace("{q}", enc)
            body, note = get(APP2 + path)
            if r.sink == "json_then_dom_innerhtml":
                # the JSON the browser fetches separately; a proxy sees this too
                aux, _ = get(f"{APP2}/s/api/echo?term={enc}")
        n += 1
        errs += bool(note)
        w.writerow([r.payload_id, r.sink, path, body, aux, note])
        if n % 1000 == 0:
            print(f"  {n} fetched, {errs} errors", flush=True)
    fh.close()
    print(f"\ndone: {n} fetched, {errs} errors -> {OUT}")


if __name__ == "__main__":
    main()
