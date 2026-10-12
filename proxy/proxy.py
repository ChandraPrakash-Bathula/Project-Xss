"""Instrumented reverse proxy :8080 -> Flask :5000.

MODE=off      pass-through, no detector
MODE=observe  score every request, log, always forward (used for all timing runs)
MODE=block    score, return 403 when score >= threshold

The detector runs inline in the event loop (no executor). This is deliberate: a slow
model then shows up as queueing delay in client-observed p95, which is what Step 8 measures.
"""
import json
import os
import sys
import time
from pathlib import Path

import aiohttp
import joblib
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import APP_URL, LOGS, MODELS, decode_once, norm  # noqa: E402

MODE = os.environ.get("MODE", "off")
MODEL_PATH = os.environ.get("MODEL", str(MODELS / "baseline.joblib"))
LOG_PATH = os.environ.get("LOG", str(LOGS / "proxy.jsonl"))
PORT = int(os.environ.get("PORT", "8080"))

bundle = joblib.load(MODEL_PATH) if MODE != "off" else None
model = bundle["pipeline"] if bundle else None
THRESH = float(os.environ.get("THRESH", bundle["threshold"] if bundle else 0.5))


async def handle(req: web.Request) -> web.Response:
    t0 = time.perf_counter()
    raw_qs = req.raw_path.partition("?")[2]
    decoded, text, score, inf_ms, blocked = None, None, None, 0.0, False
    if model is not None:
        decoded = decode_once(raw_qs)   # what Flask sees
        text = norm(decoded)            # what the model sees (same form as norm_text)
        t1 = time.perf_counter()
        score = float(model.predict_proba([text])[0, 1])
        inf_ms = (time.perf_counter() - t1) * 1000
        blocked = MODE == "block" and score >= THRESH

    req_bytes = len(f"{req.method} {req.raw_path} HTTP/1.1\r\n") + sum(
        len(k) + len(v) + 4 for k, v in req.raw_headers) + 2
    if blocked:
        status, body, ctype = 403, b"blocked", "text/plain"
    else:
        # raw_path is forwarded untouched, so Flask decodes it exactly as without the proxy
        async with app["session"].request(req.method, APP_URL + req.raw_path) as r:
            body = await r.read()
            status = r.status
            ctype = r.headers.get("Content-Type", "text/html").split(";")[0]

    app["log"].write(json.dumps({
        "ts": time.time(), "mode": MODE, "model": Path(MODEL_PATH).stem if model else None,
        "rid": req.headers.get("X-Req-Id"), "path": req.path, "raw_qs": raw_qs,
        "decoded": decoded, "text": text, "score": score, "blocked": blocked, "inference_ms": inf_ms,
        "status": status, "req_bytes": req_bytes, "resp_bytes": len(body),
        "proxy_ms": (time.perf_counter() - t0) * 1000,
    }) + "\n")
    return web.Response(body=body, status=status, content_type=ctype)


async def on_startup(app):
    # One pooled session; no total-timeout so the client decides what a timeout is.
    app["session"] = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=200), auto_decompress=False)
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    app["log"] = open(LOG_PATH, "a", buffering=1)


async def on_cleanup(app):
    await app["session"].close()
    app["log"].close()


app = web.Application(handler_args={"max_line_size": 65536, "max_field_size": 65536})
app.router.add_route("*", "/{tail:.*}", handle)
app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)

if __name__ == "__main__":
    print(f"proxy MODE={MODE} model={MODEL_PATH if model else None} thresh={THRESH} log={LOG_PATH}")
    web.run_app(app, host="127.0.0.1", port=PORT, print=None, access_log=None)
