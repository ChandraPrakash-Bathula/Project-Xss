"""Shared constants and the single definition of "what the detector sees"."""
import re
from pathlib import Path
from urllib.parse import parse_qsl, quote

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
SEED = 42

import os
APP_PORT = int(os.environ.get("APP_PORT", "5000"))
APP_URL = os.environ.get("APP_URL", f"http://127.0.0.1:{APP_PORT}")
PROXY_URL = "http://127.0.0.1:8080"
SAFE_ENDPOINT = "/reflect-text"
UNSAFE_ENDPOINTS = ["/reflect-html", "/reflect-attr", "/reflect-js", "/reflect-decode"]

# Harmless execution marker. Payloads are rewritten to set this flag instead of alert().
MARKER = "window.__xss_fired=true"
MARKER_CHECK = "window.__xss_fired === true"

FAMILIES = ["transport", "entity", "js"]
HELD_OUT_FAMILY = "js"
SEEN_FAMILIES = [f for f in FAMILIES if f != HELD_OUT_FAMILY]


def decode_once(raw_query_string: str) -> str:
    """Decode the query string ONCE, exactly like Flask/werkzeug, and join the values.

    Manifest `text` is always this once-decoded form.
    """
    return " ".join(v for _, v in parse_qsl(raw_query_string, keep_blank_values=True))


def norm(s: str) -> str:
    """Lowercase + collapse whitespace. The ONE definition of the model's input form.

    Used by data/build_dataset.py to fill `norm_text` and by the proxy at scoring time,
    so the detector sees the identical representation in training and in serving.
    """
    return re.sub(r"\s+", " ", str(s).strip().lower())


def detector_text(raw_query_string: str) -> str:
    """What the proxy feeds the model: decode once, then normalise."""
    return norm(decode_once(raw_query_string))


def url_for(endpoint: str, text: str, base: str = PROXY_URL) -> str:
    """URL whose once-decoded `q` equals `text`."""
    return f"{base}{endpoint}?q={quote(text, safe='')}"
