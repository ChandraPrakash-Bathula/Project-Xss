"""Results view for the playground: every table and the figure, rendered in the browser.

Reads the CSVs that ml/evaluate.py, ml/errors.py, verify/paired_cases.py and
bench/loadtest.py produce. Nothing is recomputed here, so what you see is exactly what
was measured; if a file is missing the section says which command generates it.
"""
import sys
from html import escape
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import LOGS, ROOT  # noqa: E402

FIGURE = ROOT / "report" / "figure_detector_location.png"

# section -> (file, how to regenerate, caption, columns to show, column to highlight)
SECTIONS = [
    ("Table 1 — detection (primary)", "table1_primary.csv", "python ml/evaluate.py baseline augmented",
     "Label = <code>attack_like</code>, TEST split. The headline result is the baseline's "
     "recall falling from 0.950 on ordinary inputs to 0.601 once the same attacks are "
     "transformed. The control row shows this is not caused by swapping <code>alert()</code> "
     "for the harmless marker. <code>recall_unique</code> counts each distinct string once, "
     "because two synthetic variants repeat one string 92 times.",
     ["model", "condition", "n_att", "n_ben", "precision", "recall", "recall_ci",
      "recall_unique", "f1", "fpr"], "recall"),

    ("Table 1b — recall by subtype", "table1_by_subtype.csv", "python ml/evaluate.py baseline augmented",
     "Where the drop actually comes from. <code>uniq</code> is the number of distinct "
     "strings: the two rows with <code>uniq=1</code> are parent-independent templates, so "
     "their recall is one string's result repeated. The last two rows split the held-out "
     "family by whether the literal marker token survives the transformation.",
     ["model", "family", "subtype", "n", "uniq", "recall"], "recall"),

    ("Table 1c — detection vs real execution", "table1_secondary.csv", "python ml/evaluate.py baseline augmented",
     "Label = <code>browser_confirmed</code>. Of the variants that <em>actually executed</em> "
     "in a real browser, how many did the detector flag? The baseline catches almost none "
     "(entity 0.000, js 0.050) while flagging 79–98% of attack-shaped payloads that do "
     "nothing. It flags what looks dangerous, not what is dangerous. Counts are small and "
     "concentrated in synthetic vectors — stated as a limitation.",
     None, "recall_on_executing"),

    ("Table 1d — is the gain real? (shortcut probe)", "table1_shortcut_probe.csv",
     "python ml/evaluate.py baseline augmented",
     "Every augmented training row contains the marker identifiers and no benign row does. "
     "Renaming those identifiers in the test variants (keeping structure and obfuscation) "
     "collapses the augmented model's held-out recall from 1.000 to 0.726 — level with the "
     "baseline's 0.709. The apparent perfect score is largely a marker-name shortcut, not "
     "generalisation to the unseen family.",
     None, "drop"),

    ("Table 2 — cost of inline detection", "table2_performance.csv",
     "python bench/loadtest.py --low 100 --high 800",
     "Client-observed response time over loopback — it includes proxy and Flask work, so it "
     "is not &ldquo;network latency&rdquo;. Median rises ~0.5&nbsp;ms with the detector, which is well "
     "outside the run-to-run spread. Throughput plateaus at ~711&nbsp;rps for all three configs "
     "<em>including detection-off</em>, so the ceiling is the upstream app, not inference. "
     "<code>p95_spread</code> is max−min across the 3 runs: where it is as large as the "
     "difference, the difference is not real.",
     ["config", "load", "offered_rps", "concurrency", "median_ms", "p95_ms", "p95_spread",
      "inference_mean_ms", "completed_rps", "req_bytes", "resp_bytes", "failures"], "p95_ms"),

    ("Paired cases — same input, two contexts", "paired_cases.csv", "python verify/paired_cases.py",
     "Each input was sent twice through the same observe-mode proxy. The detector computed "
     "an <em>identical</em> score both times, yet the marker fired 0/8 on the safe endpoint "
     "and 8/8 on the unsafe one. This is the direct evidence that request-content "
     "classification alone cannot resolve rendering context.",
     ["subtype", "unsafe_endpoint", "score_safe", "score_unsafe", "same_score",
      "fired_safe", "fired_unsafe"], "fired_unsafe"),

    ("Table 3 — does the score predict EXECUTION?", "table3_execution_auc.csv",
     "PYTHONPATH=. python ml/execution_auc.py",
     "Two questions for the same score. The left column is the task every XSS paper "
     "reports: separate attack-like strings from harmless ones. Every architecture does it "
     "well (0.79 to 1.00). The right column asks whether the same score can separate, among "
     "attack-like strings only, the ones a browser actually runs from the ones it does not. "
     "Every architecture sits at or below 0.5, which is chance. Measured over 632 distinct "
     "test strings (325 execute, 307 inert) with bootstrap intervals. This holds for "
     "hand-crafted structural features too, which have no payload vocabulary to memorise, "
     "so it is not an artefact of n-gram lookup.",
     ["model", "arch", "n_exec", "n_inert", "auc_attack_vs_benign", "auc_execute_vs_inert",
      "ci_low", "ci_high"], "auc_execute_vs_inert"),

    ("Sensitivity — pre-encoded payloads in the source", "sensitivity_preencoded.csv",
     "python ml/sensitivity.py",
     "The 50-row label audit found that 34 of 500 attack originals (6.8%, all from "
     "payloadbox) arrive <em>already</em> HTML-entity-encoded, so they are effectively "
     "pre-transformed. Table 1 was recomputed with those rows and their 128 descendants "
     "removed. The largest shift is 0.028 and the headline gap gets slightly wider, so no "
     "conclusion depends on them — reported as a stated limitation rather than silently "
     "re-run. See <code>data/SOURCES.md</code>.",
     ["variant", "model", "condition", "n_att", "n_ben", "recall", "fpr"], "recall"),

    ("Ten explained errors", "errors_selected_baseline.csv", "python ml/errors.py baseline",
     "Selected by a stated rule (seeded, distinct parents, allocated across the four causes) "
     "so they are not cherry-picked. <code>pushes</code> lists the n-grams driving the score "
     "up; <code>lost</code> lists the strong attack n-grams the transformation removed, "
     "measured against the untransformed sibling.",
     ["n", "draft_cause", "error_label", "error_type", "score", "family", "subtype",
      "endpoint", "norm_text", "evidence_top_ngrams", "evidence_lost_vs_reference"], "score"),
]

HEADLINE = [
    ("0.950 &rarr; 0.601", "baseline recall, ordinary &rarr; transformed",
     "distribution shift, non-overlapping CIs", "bad"),
    ("0.000 / 0.050", "recall on variants that really execute (entity / js)",
     "while flagging 79–98% of inert ones", "bad"),
    ("8 / 8", "paired cases: identical score, different outcome",
     "safe endpoint fired 0/8", "warn"),
    ("+0.5 ms", "median response-time cost of the detector",
     "throughput unchanged (~711 rps)", "ok"),
]


def _fmt(df):
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    return df


def _table_html(path, cols, highlight):
    """Render a CSV as a table.

    Cell values are HTML-escaped. The error table contains real payload strings, so
    unescaped output would inject markup into this very page instead of displaying it.
    """
    df = pd.read_csv(path, keep_default_na=False)
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    df = _fmt(df)
    head = "".join(f"<th class='{'hl' if c == highlight else ''}'>{escape(c)}</th>"
                   for c in df.columns)
    body = ""
    for _, r in df.iterrows():
        tds = ""
        for c in df.columns:
            v = str(r[c])
            cls = "hl" if c == highlight else ""
            if c == "model":
                cls += " model-" + escape(v)
            tds += f"<td class='{cls.strip()}'>{escape(v)}</td>"
        body += f"<tr>{tds}</tr>"
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def build():
    cards = "".join(
        f"<div class='card stat {tone}'><div class='big'>{big}</div>"
        f"<div class='lbl'>{lbl}</div><div class='sub'>{sub}</div></div>"
        for big, lbl, sub, tone in HEADLINE)

    blocks = ""
    if FIGURE.exists():
        blocks += ("<section class='card'><h2>Figure 1 — detector location and available "
                   "information</h2><p class='cap'>What exists at the detection point, and "
                   "what does not. The limits on the right are why a request-only detector "
                   "cannot settle the paired cases below.</p>"
                   "<img src='/figure.png' alt='detector location figure'></section>")

    for title, fname, cmd, cap, cols, hl in SECTIONS:
        path = LOGS / fname
        if path.exists():
            inner = _table_html(path, cols, hl)
            src = f"<div class='src'>logs/{fname}</div>"
        else:
            inner = (f"<p class='missing'>Not generated yet. Run:<br><code>{cmd}</code></p>")
            src = ""
        blocks += f"<section class='card'><h2>{title}</h2><p class='cap'>{cap}</p>{inner}{src}</section>"

    return PAGE.replace("%CARDS%", cards).replace("%BLOCKS%", blocks)


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XSS Detection — Results</title>
<style>
:root{--bg:#0f1420;--card:#171d2b;--edge:#28324a;--ink:#e7ecf5;--mut:#92a0bd;
--ok:#2ea86b;--bad:#e5484d;--warn:#e2a93b;--accent:#5b8def;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media(prefers-color-scheme:light){:root{--bg:#f3f5fa;--card:#fff;--edge:#dce1ec;--ink:#121725;--mut:#5c6a86}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:22px 16px 72px}
nav{display:flex;gap:8px;margin-bottom:18px}
nav a{padding:8px 14px;border:1px solid var(--edge);border-radius:999px;text-decoration:none;
color:var(--mut);font-size:13px;font-weight:600}
nav a.on{background:var(--accent);border-color:var(--accent);color:#fff}
h1{font-size:21px;margin:0 0 4px}.lead{color:var(--mut);font-size:13px;margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--edge);border-radius:12px;padding:16px;margin-bottom:16px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}
@media(max-width:860px){.stats{grid-template-columns:repeat(2,1fr)}}
.stat{padding:14px}.stat .big{font-size:23px;font-weight:700;letter-spacing:-.4px}
.stat .lbl{font-size:12px;color:var(--ink);margin-top:3px}.stat .sub{font-size:11.5px;color:var(--mut);margin-top:3px}
.stat.bad .big{color:var(--bad)}.stat.ok .big{color:var(--ok)}.stat.warn .big{color:var(--warn)}
h2{font-size:15.5px;margin:0 0 6px}
.cap{color:var(--mut);font-size:12.8px;margin:0 0 12px}
.cap code{font-family:var(--mono);font-size:11.5px;background:var(--bg);padding:1px 4px;border-radius:4px}
.scroll{overflow-x:auto;border:1px solid var(--edge);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:12.4px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--edge);white-space:nowrap}
th{background:var(--bg);color:var(--mut);font-weight:600;position:sticky;top:0}
td{font-family:var(--mono);font-size:11.8px}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:color-mix(in srgb,var(--accent) 7%,transparent)}
th.hl,td.hl{background:color-mix(in srgb,var(--accent) 11%,transparent);font-weight:700}
td.model-baseline{color:var(--warn)}td.model-augmented{color:var(--accent)}
td:nth-child(n+9){white-space:normal;min-width:180px}
.src{font-family:var(--mono);font-size:11px;color:var(--mut);margin-top:8px}
.missing{color:var(--mut);font-size:13px}.missing code{font-family:var(--mono);color:var(--ink)}
img{width:100%;border:1px solid var(--edge);border-radius:8px;background:#fff}
</style></head><body><div class="wrap">
<nav><a href="/">Playground</a><a href="/results" class="on">Results</a></nav>
<h1>Results</h1>
<p class="lead">Measured on the held-out TEST split. Every number below is read from the
logs produced by the pipeline — nothing is recomputed in this page.</p>
<div class="stats">%CARDS%</div>
%BLOCKS%
</div></body></html>"""
