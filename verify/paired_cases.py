"""Step 3: paired rendering-context cases (local, marker-only).

Each input is sent twice through an OBSERVE-mode proxy: once to the safe /reflect-text
endpoint and once to an unsafe endpoint. The detector scores the identical query string
both times (same score), but the browser outcome differs. Request-content classification
alone cannot tell these apart, because the difference lives in how the application uses
the value, which the proxy never sees.

Starts its own observe proxy on :8081 (baseline model) so it does not disturb :8080.
Needs the Flask app on :5000.   ->  logs/paired_cases.csv
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import DATA, LOGS, MARKER_CHECK, SAFE_ENDPOINT  # noqa: E402

PORT = 8081
BASE = f"http://127.0.0.1:{PORT}"
PLOG = LOGS / "paired_proxy.jsonl"
N_CASES = 8


def pick_cases(m):
    """Confirmed-executing on load, one per distinct string, spread over unsafe endpoints."""
    ok = m[(m.label_kind == "attack_variant") & (m.browser_confirmed == "1")]
    ok = ok.drop_duplicates("norm_text").sample(frac=1, random_state=42)
    per_ep = ok.groupby("endpoint").head(2)            # variety across contexts first
    return pd.concat([per_ep, ok]).drop_duplicates("id").head(N_CASES)


def fired(page, url):
    page.goto(url, wait_until="load", timeout=5000)
    page.wait_for_timeout(200)
    return bool(page.evaluate(MARKER_CHECK))


def main():
    m = pd.read_csv(DATA / "manifest.csv", keep_default_na=False)
    cases = pick_cases(m)

    PLOG.unlink(missing_ok=True)
    env = dict(os.environ, MODE="observe", PORT=str(PORT), LOG=str(PLOG),
               MODEL=str(ROOT / "models" / "baseline.joblib"))
    proxy = subprocess.Popen([sys.executable, str(ROOT / "proxy" / "proxy.py")], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):                               # wait for the proxy to come up
            try:
                urlopen(f"{BASE}{SAFE_ENDPOINT}?q=ping", timeout=1); break
            except Exception:
                time.sleep(0.25)

        rows = []
        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context()
            for _, r in cases.iterrows():
                res = {}
                for side, ep in [("safe", SAFE_ENDPOINT), ("unsafe", r.endpoint)]:
                    page = ctx.new_page()
                    page.on("dialog", lambda d: d.dismiss())
                    res[side] = fired(page, f"{BASE}{ep}?q={r.wire_q}")
                    page.close()
                rows.append(dict(id=r.id, subtype=r.subtype, input=r.norm_text, wire_q=r.wire_q,
                                 safe_endpoint=SAFE_ENDPOINT, unsafe_endpoint=r.endpoint,
                                 fired_safe=int(res["safe"]), fired_unsafe=int(res["unsafe"])))
            browser.close()
    finally:
        proxy.terminate(); proxy.wait()

    # scores come from the proxy's own log: what the detector actually computed in-line
    log = [json.loads(l) for l in PLOG.read_text().splitlines() if l.strip()]
    score = {}
    for e in log:
        if e["score"] is not None and e["raw_qs"] != "q=ping":
            score.setdefault((e["path"], e["raw_qs"]), e["score"])
    out = pd.DataFrame(rows)
    out["score_safe"] = [score.get((SAFE_ENDPOINT, f"q={w}")) for w in out.wire_q]
    out["score_unsafe"] = [score.get((ep, f"q={w}")) for ep, w in zip(out.unsafe_endpoint, out.wire_q)]
    out["same_score"] = (out.score_safe - out.score_unsafe).abs() < 1e-12
    out.to_csv(LOGS / "paired_cases.csv", index=False)

    for _, r in out.iterrows():
        print(f"\n{r.subtype}:  {r.input[:70]}")
        print(f"   {r.safe_endpoint:<15} score={r.score_safe:.3f}  fired={r.fired_safe}")
        print(f"   {r.unsafe_endpoint:<15} score={r.score_unsafe:.3f}  fired={r.fired_unsafe}")
    print(f"\nidentical score in every pair: {bool(out.same_score.all())}   "
          f"safe fired: {out.fired_safe.sum()}/{len(out)}   unsafe fired: {out.fired_unsafe.sum()}/{len(out)}")
    print(f"-> {LOGS / 'paired_cases.csv'}")


if __name__ == "__main__":
    main()
