"""Per vulnerability class: can a payload-only model predict execution, and does the
detector even HAVE the payload at the moment that decides it?

The brief scopes the main experiment to reflected XSS and asks for stored and DOM-based
XSS to be discussed as visibility limitations. App 2 implements all three, so they can be
measured rather than discussed.

Two separate questions, and conflating them is the mistake this script avoids:

  1. PREDICTABILITY  given the payload string, how well can execution be predicted in
                     this class? (train on execution, payload only, app 2 splits)
  2. AVAILABILITY    at the moment the payload reaches the browser, is it present in the
                     request a proxy would inspect?

A class can score high on (1) and still be undefendable, because (2) is zero. That is
exactly what happens for stored and DOM XSS, and it is an information argument, not a
modelling one.

    PYTHONPATH=. python ml/visibility_by_class.py   (needs app 2 on :5002)
    -> logs/table10_visibility_by_class.csv
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import LOGS, SEED  # noqa: E402

APP2 = "http://127.0.0.1:5002"
MARKER_PAYLOAD = "<img src=x onerror=window.__xss_fired=true>"
CLS = {"html_text_escaped": "reflected_escaped", "html_body_raw": "reflected_raw",
       "attribute_double_quoted": "reflected_attr", "js_string_literal": "reflected_js",
       "html_body_partial_sanitise": "reflected_filtered",
       "json_then_dom_innerhtml": "DOM_client_sink",
       "stored_html_body_raw": "STORED_post_then_get"}


def predictability():
    d = pd.read_csv(LOGS / "cross_context_app2.csv", keep_default_na=False)
    d = d[d.note == ""].copy()
    d["ex"] = ((d.fired_load == 1) | (d.fired_interact == 1)).astype(int)
    d["cls"] = d.sink.map(CLS)
    tr, te = d[d.split.isin(["train", "val"])], d[d.split == "test"]
    assert not (set(tr.group_id) & set(te.group_id)), "group leak"
    model = Pipeline([("v", TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                            sublinear_tf=True, min_df=2)),
                      ("c", LogisticRegression(max_iter=3000, random_state=SEED))])
    model.fit(tr.norm_text.tolist(), tr.ex.values)
    out = {}
    for c, g in te.groupby("cls"):
        auc = (roc_auc_score(g.ex.values, model.predict_proba(g.norm_text.tolist())[:, 1])
               if g.ex.nunique() == 2 else float("nan"))
        out[c] = dict(n=len(g), n_exec=int(g.ex.sum()), rate=float(g.ex.mean()), auc=auc)
    return out


def availability():
    """Empirically check, per class, whether the payload is in the deciding request."""
    enc = urllib.parse.quote(MARKER_PAYLOAD, safe="")
    res = {}

    # reflected: payload is in the very request that renders it
    body = urllib.request.urlopen(f"{APP2}/s/profile?term={enc}", timeout=5).read().decode()
    res["reflected_raw"] = dict(in_request=1, in_response=int("onerror" in body))

    # DOM: the server hands back a page with no payload; a separate fetch supplies it and
    # client script writes it with innerHTML
    body = urllib.request.urlopen(f"{APP2}/s/client?term={enc}", timeout=5).read().decode()
    res["DOM_client_sink"] = dict(in_request=1, in_response=int("onerror" in body))

    # stored: the payload arrives in a POST, and is delivered by a LATER, different request
    data = urllib.parse.urlencode({"term": MARKER_PAYLOAD}).encode()
    nid = json.load(urllib.request.urlopen(
        urllib.request.Request(APP2 + "/s/note", data=data), timeout=5))["id"]
    get_url = f"/s/note/{nid}"
    body = urllib.request.urlopen(APP2 + get_url, timeout=5).read().decode()
    res["STORED_post_then_get"] = dict(in_request=int("onerror" in get_url),
                                       in_response=int("onerror" in body))
    return res


def main():
    try:
        urllib.request.urlopen(APP2 + "/s/sinks", timeout=3)
    except Exception:
        sys.exit("app 2 not running: ./venv/bin/python app/vuln_app2.py &")

    pred, avail = predictability(), availability()
    rows = []
    for c, p in sorted(pred.items()):
        a = avail.get(c, {})
        rows.append(dict(vuln_class=c, n=p["n"], n_exec=p["n_exec"],
                         exec_rate=round(p["rate"], 3),
                         auc_payload_only=round(p["auc"], 3) if p["auc"] == p["auc"] else None,
                         payload_in_deciding_request=a.get("in_request"),
                         payload_in_response=a.get("in_response")))
    t = pd.DataFrame(rows)
    t.to_csv(LOGS / "table10_visibility_by_class.csv", index=False)
    print(t.to_string(index=False))
    print("\nPREDICTABILITY vs AVAILABILITY")
    print("-" * 72)
    print("  reflected_raw         payload IS in the request that renders it -> defendable")
    print("  DOM_client_sink       payload is in the request, but NOT in the server")
    print("                        response; the sink is client-side script")
    print("  STORED_post_then_get  the GET that delivers the payload does NOT contain it.")
    print("                        A request-only detector has ZERO signal at that moment,")
    print("                        no matter how good the model is.")
    print(f"\n-> {LOGS / 'table10_visibility_by_class.csv'}")


if __name__ == "__main__":
    main()
