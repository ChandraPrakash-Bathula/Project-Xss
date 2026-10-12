"""Does the context finding hold on a SECOND application? And does it transfer?

ml/context_crossed.py measured the value of the output context on app 1. The obvious
objection is that the result is a property of one application's templates. App 2 is
written differently on purpose: a templating engine with autoescaping, a partial
sanitiser that strips only "<script", a JSON endpoint whose output the browser writes
with innerHTML, and a stored path where the payload arrives in a POST and is delivered
by a later GET.

Three questions:
  1. within app 2    does payload + sink still beat payload alone?
  2. cross-app       train on app 1, test on app 2. This is the real generalisation
                     test, and the one a reviewer will ask about.
  3. new sinks       app 2 has three sink types app 1 cannot express. Are they harder?

    PYTHONPATH=. python ml/context_app2.py  ->  logs/table7_app2_context.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import LOGS, SEED  # noqa: E402

A1 = LOGS / "cross_context.csv"
A2 = LOGS / "cross_context_app2.csv"
OUT = LOGS / "table7_app2_context.csv"

# sinks app 2 has and app 1 does not
NEW_SINKS = {"html_body_partial_sanitise", "json_then_dom_innerhtml", "stored_html_body_raw"}


def load(path):
    d = pd.read_csv(path, keep_default_na=False)
    d = d[d.note == ""].copy()
    d["ex"] = ((d.fired_load == 1) | (d.fired_interact == 1)).astype(int)
    return d


def build():
    return Pipeline([("v", TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                           sublinear_tf=True, min_df=2)),
                     ("c", LogisticRegression(max_iter=3000, random_state=SEED))])


def fmt(d, kind):
    if kind == "payload_only":
        return d.norm_text.tolist()
    if kind == "sink_only":
        return d.sink.tolist()
    return [f"[{s}] {t}" for s, t in zip(d.sink, d.norm_text)]


def boot(y, s, groups, n=600, seed=SEED):
    rng = np.random.default_rng(seed)
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n):
        take = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[take])) == 2:
            out.append(roc_auc_score(y[take], s[take]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan,) * 2


def run(tr, te, kind, label, rows):
    if te.ex.nunique() < 2:
        return
    pipe = build().fit(fmt(tr, kind), tr.ex.values)
    s = pipe.predict_proba(fmt(te, kind))[:, 1]
    auc = roc_auc_score(te.ex.values, s)
    lo, hi = boot(te.ex.values, s, te.group_id.values)
    rows.append(dict(setting=label, input=kind, auc=auc, ci_low=lo, ci_high=hi,
                     avg_precision=average_precision_score(te.ex.values, s),
                     n_train=len(tr), n_test=len(te), test_exec=int(te.ex.sum())))
    print(f"  {label:<28} {kind:<14} AUC {auc:.3f} [{lo:.3f},{hi:.3f}]  "
          f"AP {rows[-1]['avg_precision']:.3f}")


def main():
    a1, a2 = load(A1), load(A2)
    print(f"app1 {len(a1)} obs / {a1.payload_id.nunique()} payloads / {a1.sink.nunique()} sinks")
    print(f"app2 {len(a2)} obs / {a2.payload_id.nunique()} payloads / {a2.sink.nunique()} sinks")
    print(f"sinks only in app2: {sorted(NEW_SINKS)}\n")

    rows = []
    print("1. WITHIN APP 2 (its own train/test split)")
    tr2, te2 = a2[a2.split.isin(["train", "val"])], a2[a2.split == "test"]
    assert not (set(tr2.group_id) & set(te2.group_id)), "group leak within app2"
    for k in ("payload_only", "sink_only", "payload_sink"):
        run(tr2, te2, k, "app2 -> app2", rows)

    print("\n2. CROSS-APP (train on ALL of app 1, test on ALL of app 2)")
    for k in ("payload_only", "sink_only", "payload_sink"):
        # sink_only cannot transfer: app2's three new sink labels were never seen
        run(a1, a2, k, "app1 -> app2", rows)

    print("\n3. APP 2's NEW SINKS ONLY (unseen sink types, trained on app 1)")
    new = a2[a2.sink.isin(NEW_SINKS)]
    for k in ("payload_only", "payload_sink"):
        run(a1, new, k, "app1 -> app2 new sinks", rows)

    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False)

    print("\n" + "=" * 78)
    for label in t.setting.unique():
        g = t[t.setting == label]
        if {"payload_only", "payload_sink"} <= set(g.input):
            p = g.loc[g.input == "payload_only", "auc"].iloc[0]
            s = g.loc[g.input == "payload_sink", "auc"].iloc[0]
            print(f"  {label:<28} context worth {s - p:+.3f} AUC")
    print("\n  app1 -> app1 reference (ml/context_crossed.py): +0.157")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
