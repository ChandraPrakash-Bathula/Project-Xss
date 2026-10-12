"""Step 5: communication and response-time cost of inline detection.

Three proxy configurations, all reaching the SAME endpoints so the comparison is fair:
    A_off        MODE=off      no detector
    B_baseline   MODE=observe  baseline model scored inline, always forwarded
    C_augmented  MODE=observe  augmented model scored inline, always forwarded

Observe mode is used for every timing run (never block), so no request is short-circuited
by a 403 and all three configs do the same upstream work. Blocking decisions are evaluated
separately in ml/evaluate.py.

Two offered loads x three repetitions, config order alternated between rounds. A fixed
request mix (same file, same order, seeded) is replayed every run. The first WARMUP
requests of each run are discarded.

Reported per run: client-observed median and p95 response time, completed requests/s,
proxy inference time (mean/p95, from the proxy's own log), HTTP request/response bytes,
failures and timeouts.

NOTE the metric name: this is client-observed RESPONSE TIME over loopback. It includes
proxy and Flask work. It is not "network latency".

    python bench/loadtest.py --calibrate     # find a saturating high load first
    python bench/loadtest.py                 # full 3 configs x 2 loads x 3 runs
"""
import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import aiohttp
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import DATA, LOGS, SEED  # noqa: E402

PORT = 8090                      # dedicated port, leaves :8080 untouched
BASE = f"http://127.0.0.1:{PORT}"
MIX_CSV = LOGS / "request_mix.csv"
WARMUP = 200
N_REQUESTS = 1800
ATTACK_SHARE = 0.10              # real traffic is mostly benign; stated in the report
REQ_TIMEOUT = 10.0

CONFIGS = {
    "A_off": dict(MODE="off"),
    "B_baseline": dict(MODE="observe", MODEL=str(ROOT / "models" / "baseline.joblib")),
    "C_augmented": dict(MODE="observe", MODEL=str(ROOT / "models" / "augmented.joblib")),
}


# ----------------------------------------------------------------- request mix
def build_mix():
    """Fixed mix drawn from the TEST split only, fixed order, fixed seed."""
    m = pd.read_csv(DATA / "manifest.csv", keep_default_na=False)
    test = m[m.split == "test"]

    att = test[(test.label_attack_like == 1) & (test.label_kind == "attack_variant")]
    ben = test[test.label_attack_like == 0]
    n_att = int(N_REQUESTS * ATTACK_SHARE)
    n_ben = N_REQUESTS - n_att

    att = att.sample(n=n_att, replace=len(att) < n_att, random_state=SEED)
    ben = ben.sample(n=n_ben, replace=len(ben) < n_ben, random_state=SEED)

    rows = []
    for _, r in pd.concat([att, ben]).sample(frac=1, random_state=SEED).iterrows():
        wire = r.wire_q if r.wire_q else quote(r.text, safe="")
        ep = r.endpoint if r.endpoint else "/reflect-html"
        rows.append({"endpoint": ep, "wire_q": wire, "attack_like": r.label_attack_like})
    mix = pd.DataFrame(rows)
    MIX_CSV.parent.mkdir(exist_ok=True)
    mix.to_csv(MIX_CSV, index=False)
    return mix


# ----------------------------------------------------------------- proxy control
def start_proxy(cfg, log_path):
    env = dict(os.environ, PORT=str(PORT), LOG=str(log_path), **cfg)
    p = subprocess.Popen([sys.executable, str(ROOT / "proxy" / "proxy.py")], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(80):
        try:
            urlopen(f"{BASE}/reflect-text?q=ping", timeout=1)
            return p
        except Exception:
            time.sleep(0.25)
    p.terminate()
    raise SystemExit(f"proxy failed to start for {cfg}")


# ----------------------------------------------------------------- load generator
async def worker(session, queue, out):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        i, ep, wire, due = item
        now = time.perf_counter()
        if due > now:
            await asyncio.sleep(due - now)          # open-loop: hold the offered rate
        t0 = time.perf_counter()
        status, nbytes, err = 0, 0, ""
        try:
            async with session.get(f"{BASE}{ep}?q={wire}") as r:
                body = await r.read()
                status, nbytes = r.status, len(body)
        except asyncio.TimeoutError:
            err = "timeout"
        except Exception as e:
            err = type(e).__name__
        out.append((i, (time.perf_counter() - t0) * 1000, status, nbytes, err))
        queue.task_done()


async def run_once(mix, rate, concurrency):
    out = []
    queue = asyncio.Queue(maxsize=concurrency * 4)
    timeout = aiohttp.ClientTimeout(total=REQ_TIMEOUT)
    conn = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        tasks = [asyncio.create_task(worker(session, queue, out)) for _ in range(concurrency)]
        start = time.perf_counter()
        for i, r in enumerate(mix.itertuples()):
            await queue.put((i, r.endpoint, r.wire_q, start + i / rate))
        for _ in tasks:
            await queue.put(None)
        await queue.join()
        wall = time.perf_counter() - start
        await asyncio.gather(*tasks)
    return out, wall


# ----------------------------------------------------------------- measurement
def summarise(out, wall, proxy_log, config, load_name, rate, concurrency, rep):
    out = sorted(out)[WARMUP:]                       # discard warm-up by request index
    lat = [o[1] for o in out if not o[4]]
    fails = sum(1 for o in out if o[4] or o[2] != 200)
    timeouts = sum(1 for o in out if o[4] == "timeout")
    resp_bytes = [o[3] for o in out if not o[4]]

    inf, req_b = [], []
    if proxy_log.exists():
        entries = [json.loads(l) for l in proxy_log.read_text().splitlines() if l.strip()]
        entries = [e for e in entries if e.get("raw_qs") != "q=ping"][WARMUP:]
        inf = [e["inference_ms"] for e in entries if e.get("score") is not None]
        req_b = [e["req_bytes"] for e in entries]

    def pct(v, q):
        return statistics.quantiles(v, n=100)[q - 1] if len(v) > 1 else (v[0] if v else float("nan"))

    measured = len(out)
    return {
        "config": config, "load": load_name, "offered_rps": rate, "concurrency": concurrency,
        "rep": rep, "n_measured": measured,
        "median_ms": statistics.median(lat) if lat else float("nan"),
        "p95_ms": pct(lat, 95),
        "completed_rps": measured / wall if wall else float("nan"),
        "inference_mean_ms": statistics.fmean(inf) if inf else 0.0,
        "inference_p95_ms": pct(inf, 95) if inf else 0.0,
        "req_bytes_mean": statistics.fmean(req_b) if req_b else float("nan"),
        "resp_bytes_mean": statistics.fmean(resp_bytes) if resp_bytes else float("nan"),
        "failures": fails, "timeouts": timeouts,
    }


def one_run(mix, config, cfg, load_name, rate, concurrency, rep):
    log_path = LOGS / f"timing_{config}_{load_name}_r{rep}.jsonl"
    log_path.unlink(missing_ok=True)
    proxy = start_proxy(cfg, log_path)
    try:
        out, wall = asyncio.run(run_once(mix, rate, concurrency))
    finally:
        proxy.terminate()
        proxy.wait()
    row = summarise(out, wall, log_path, config, load_name, rate, concurrency, rep)
    print(f"  {config:<12} {load_name:<8} rep{rep}  median={row['median_ms']:.1f}ms  "
          f"p95={row['p95_ms']:.1f}ms  rps={row['completed_rps']:.0f}  "
          f"inf={row['inference_mean_ms']:.2f}ms  fail={row['failures']}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true",
                    help="probe rising rates on the baseline config to choose the high load")
    ap.add_argument("--low", type=int, default=100)
    ap.add_argument("--high", type=int, default=0, help="0 = use the calibrated default")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    mix = build_mix()
    print(f"request mix: {len(mix)} requests, {mix.attack_like.mean():.0%} attack-like, "
          f"endpoints={mix.endpoint.nunique()}  -> {MIX_CSV}")

    if args.calibrate:
        print("\ncalibrating on B_baseline (looking for p95 to climb):")
        for rate in (100, 200, 400, 600, 800):
            one_run(mix, "B_baseline", CONFIGS["B_baseline"], "cal", rate, max(8, rate // 12), 1)
        return

    high = args.high or 600
    loads = [("moderate", args.low, 16), ("high", high, 64)]
    rows = []
    names = list(CONFIGS)
    for rep in range(1, args.reps + 1):
        order = names if rep % 2 else names[::-1]        # alternate config order per round
        print(f"\nround {rep}  (order: {', '.join(order)})")
        for config in order:
            for load_name, rate, conc in loads:
                rows.append(one_run(mix, config, CONFIGS[config], load_name, rate, conc, rep))

    raw = pd.DataFrame(rows)
    raw.to_csv(LOGS / "timing_raw.csv", index=False)

    agg = raw.groupby(["config", "load"]).agg(
        offered_rps=("offered_rps", "first"), concurrency=("concurrency", "first"),
        median_ms=("median_ms", "mean"), median_spread=("median_ms", lambda s: s.max() - s.min()),
        p95_ms=("p95_ms", "mean"), p95_spread=("p95_ms", lambda s: s.max() - s.min()),
        inference_mean_ms=("inference_mean_ms", "mean"), inference_p95_ms=("inference_p95_ms", "mean"),
        completed_rps=("completed_rps", "mean"),
        req_bytes=("req_bytes_mean", "mean"), resp_bytes=("resp_bytes_mean", "mean"),
        failures=("failures", "sum"), timeouts=("timeouts", "sum")).reset_index()
    agg.to_csv(LOGS / "table2_performance.csv", index=False)

    print("\n" + "=" * 100)
    print("TABLE 2 — client-observed response time (loopback; includes proxy + Flask work)")
    print("mean over %d runs, 'spread' = max-min across runs" % args.reps)
    print("=" * 100)
    print(agg.to_string(index=False, float_format="{:.2f}".format))
    print(f"\n-> {LOGS/'timing_raw.csv'}  and  {LOGS/'table2_performance.csv'}")


if __name__ == "__main__":
    main()
