"""Second vulnerable test app, on :5002. Local only. Do not deploy.

App 1 (vuln_app.py) is a bare reflector: the parameter goes straight into one fixed
template. Real applications do not look like that, and a reviewer's first objection to a
single-application result is that the finding is a property of that one template.

This app differs in the ways that matter to a request-side detector:

  * a templating engine with autoescaping ON, so the SAFE path is safe for a structural
    reason rather than because we remembered to call escape()
  * partial sanitisation that strips "<script" but nothing else, which is the most common
    real-world half-measure and lets through every handler-based payload
  * a JSON API whose output is injected into the DOM by client-side script, so the sink is
    in the browser and the server never renders the payload at all
  * a different parameter name (term, not q) and an extra wrapper parameter, so any
    detector keyed on the literal shape of app 1's query string is exposed
  * a stored path: POST to /note then GET /note/<id>, so the request that carries the
    payload and the response that delivers it are different transactions

Every sink is annotated so the context-conditioned experiment can read it directly.
"""
import html
import re
import uuid

from flask import Flask, Response, jsonify, request
from markupsafe import escape

app = Flask(__name__)
app.jinja_env.autoescape = True

NOTES = {}

# endpoint -> the output context the parameter lands in. Consumed by ml/context.py.
SINKS = {
    "/s/search": "html_text_escaped",
    "/s/profile": "html_body_raw",
    "/s/filter": "attribute_double_quoted",
    "/s/config": "js_string_literal",
    "/s/preview": "html_body_partial_sanitise",
    "/s/api/echo": "json_then_dom_innerhtml",
    "/s/note": "stored_html_body_raw",
}

PAGE = """<!doctype html><html><head><title>Notes</title></head>
<body><div id="app"><h1>Notes</h1>{body}</div></body></html>"""


def page(body):
    return PAGE.format(body=body)


@app.get("/s/search")                      # SAFE: escaped by the template layer
def s_search():
    return page(f"<p>Results for {escape(request.args.get('term', ''))}</p>")


@app.get("/s/profile")                     # UNSAFE: raw into the body
def s_profile():
    return page("<div class='bio'>" + request.args.get("term", "") + "</div>")


@app.get("/s/filter")                      # UNSAFE: inside a quoted attribute
def s_filter():
    return page("<input name='f' value=\"" + request.args.get("term", "") + "\">")


@app.get("/s/config")                      # UNSAFE: inside a JS string literal
def s_config():
    return page("<script>var cfg={term:'" + request.args.get("term", "") + "'};</script>")


@app.get("/s/preview")                     # UNSAFE: the classic half-measure
def s_preview():
    raw = request.args.get("term", "")
    cleaned = re.sub(r"(?i)<\s*script", "", raw)   # strips one token, nothing else
    return page("<div class='preview'>" + cleaned + "</div>")


@app.get("/s/api/echo")                    # server returns JSON; the BROWSER builds the DOM
def s_api_echo():
    return jsonify(term=request.args.get("term", ""))


@app.get("/s/client")                      # the page that consumes that JSON via innerHTML
def s_client():
    return page("""<div id="out"></div><script>
      const t = new URLSearchParams(location.search).get('term') || '';
      fetch('/s/api/echo?term=' + encodeURIComponent(t))
        .then(r => r.json())
        .then(d => { document.getElementById('out').innerHTML = d.term; });
    </script>""")


@app.post("/s/note")                       # stored: request now, delivery later
def s_note_create():
    nid = uuid.uuid4().hex[:12]
    NOTES[nid] = request.form.get("term", request.args.get("term", ""))
    return jsonify(id=nid, url=f"/s/note/{nid}")


@app.get("/s/note/<nid>")                  # UNSAFE: stored value, raw into the body
def s_note_read(nid):
    return page("<article>" + NOTES.get(nid, "(not found)") + "</article>")


@app.get("/s/sinks")                       # machine-readable map for the experiments
def s_sinks():
    return jsonify(SINKS)


@app.get("/")
def index():
    links = "".join(f"<li><code>{html.escape(k)}</code> &rarr; {v}</li>" for k, v in SINKS.items())
    return page(f"<p>Local test app 2. Endpoints:</p><ul>{links}</ul>")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, threaded=True)
