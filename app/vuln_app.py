"""Intentionally vulnerable test app. Binds to 127.0.0.1 only. Do not deploy.

Each endpoint inserts the same `q` parameter into a different context.
"""
from urllib.parse import unquote

from flask import Flask, request
from markupsafe import escape

app = Flask(__name__)


def wrap(body):
    return f"<!doctype html><html><body>{body}</body></html>"


@app.get("/reflect-text")  # SAFE: HTML-escaped
def text():
    return wrap(f"<p>You searched: {escape(request.args.get('q', ''))}</p>")


@app.get("/reflect-html")  # UNSAFE: raw in HTML body
def html():
    return wrap("<p>You searched: " + request.args.get("q", "") + "</p>")


@app.get("/reflect-attr")  # UNSAFE: inside a double-quoted attribute
def attr():
    return wrap('<input value="' + request.args.get("q", "") + '">')


@app.get("/reflect-js")  # UNSAFE: inside a JS string literal
def js():
    return wrap('<script>var q = "' + request.args.get("q", "") + '";</script>')


@app.get("/reflect-decode")  # UNSAFE: app URL-decodes a second time, then raw HTML
def decode():
    return wrap("<p>You searched: " + unquote(request.args.get("q", "")) + "</p>")


if __name__ == "__main__":
    import os
    app.run(host="127.0.0.1", port=int(os.environ.get("APP_PORT", "5000")), threaded=True)
