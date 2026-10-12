"""Context-conditioned detection: how much of the execution gap does the sink close?

Table 3 shows that a detector's attack-likeness score does not predict whether a payload
executes. That is a negative result about the *score*. The constructive question is
whether execution is predictable at all from what a proxy could plausibly know, and how
much of it comes from the payload versus from the output context.

Four models, all predicting EXECUTION (not attack-likeness), all on the same group-split:

  payload_only   the request value alone. This is what a request-side detector sees.
  sink_only      the output context alone, payload discarded. An upper bound on what you
                 get from knowing the endpoint and nothing about the input.
  payload_sink   both, with the sink prepended as a token.
  payload_x_sink both, plus explicit interaction features (payload signals crossed with
                 the sink), since a plain concatenation cannot express "this payload shape
                 matters only in this context".

Reported as AUC and average precision for execution, with group-bootstrap intervals.
A large gap between payload_only and payload_sink is the measured value of context, and
is the part a request-only detector structurally cannot have.

    PYTHONPATH=. python ml/context.py   ->  logs/table4_context.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, LOGS, SEED  # noqa: E402
from ml.zoo import SparseHandcrafted  # noqa: E402

# Output context each endpoint puts the value into. App 2 publishes the same map at
# /s/sinks; app 1's is declared here because the app is a fixed fixture.
SINK = {
    "/reflect-text": "html_text_escaped",
    "/reflect-html": "html_body_raw",
    "/reflect-attr": "attribute_double_quoted",
    "/reflect-js": "js_string_literal",
    "/reflect-decode": "html_body_raw_double_decode",
}


def load():
    m = pd.read_csv(DATA / "manifest.csv", keep_default_na=False)
    a = m[(m.label_kind == "attack_variant") & (m.endpoint != "")].copy()
    wi = a.browser_confirmed_with_interaction.astype(str)
    a["ex"] = ((a.browser_confirmed.astype(str) == "1") | (wi == "1")).astype(int)
    a["sink"] = a.endpoint.map(SINK).fillna("unknown")
    # one row per (string, sink): the same payload in two contexts is two observations,
    # but the same payload repeated in one context is not new information
    a = a.drop_duplicates(["norm_text", "sink"])
    return a


def char_tfidf():
    return TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True, min_df=2)


def build(kind):
    if kind == "payload_only":
        return Pipeline([("v", char_tfidf()),
                         ("c", LogisticRegression(max_iter=2000, random_state=SEED))])
    if kind == "sink_only":
        return Pipeline([("v", TfidfVectorizer(analyzer="word", token_pattern=r"\S+")),
                         ("c", LogisticRegression(max_iter=2000, random_state=SEED))])
    if kind == "payload_sink":
        return Pipeline([("v", char_tfidf()),
                         ("c", LogisticRegression(max_iter=2000, random_state=SEED))])
    if kind == "payload_x_sink":
        return Pipeline([
            ("u", FeatureUnion([("char", char_tfidf()), ("hand", SparseHandcrafted())])),
            ("c", LogisticRegression(max_iter=3000, random_state=SEED))])
    raise ValueError(kind)


def inputs(kind, df):
    if kind == "payload_only":
        return df.norm_text.tolist()
    if kind == "sink_only":
        return df.sink.tolist()
    if kind == "payload_sink":
        # sink as a literal token the char n-grams can pick up
        return [f"[{s}] {t}" for s, t in zip(df.sink, df.norm_text)]
    if kind == "payload_x_sink":
        # cross every payload n-gram with the sink by namespacing the text per context,
        # so "this shape matters here but not there" is expressible
        return [f"[{s}] " + " ".join(f"{s}⁣{tok}" for tok in t.split()) + f" {t}"
                for s, t in zip(df.sink, df.norm_text)]
    raise ValueError(kind)


def boot(y, s, groups, fn, n=600, seed=SEED):
    rng = np.random.default_rng(seed)
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n):
        take = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[take])) == 2:
            out.append(fn(y[take], s[take]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan,) * 2


def main():
    a = load()
    tr, te = a[a.split.isin(["train", "val"])], a[a.split == "test"]
    assert not (set(tr.group_id) & set(te.group_id)), "group leak between train and test"

    print(f"rows: train+val {len(tr)}, test {len(te)}")
    print(f"test executes {te.ex.sum()} / inert {(1 - te.ex).sum()}")
    print(f"sinks present: {sorted(a.sink.unique())}\n")
    print(f"{'model':<16}{'AUC':>8}{'95% CI':>18}{'AvgPrec':>10}   what it sees")
    print("-" * 78)

    desc = {"payload_only": "request value only (a proxy's view)",
            "sink_only": "output context only, no payload",
            "payload_sink": "value + sink token",
            "payload_x_sink": "value + sink + interactions"}
    rows = []
    for kind in ["payload_only", "sink_only", "payload_sink", "payload_x_sink"]:
        pipe = build(kind).fit(inputs(kind, tr), tr.ex.values)
        s = pipe.predict_proba(inputs(kind, te))[:, 1]
        auc = roc_auc_score(te.ex.values, s)
        ap = average_precision_score(te.ex.values, s)
        lo, hi = boot(te.ex.values, s, te.group_id.values, roc_auc_score)
        rows.append(dict(model=kind, auc=auc, ci_low=lo, ci_high=hi, avg_precision=ap,
                         n_train=len(tr), n_test=len(te), sees=desc[kind]))
        print(f"{kind:<16}{auc:>8.3f}   [{lo:.3f}, {hi:.3f}]{ap:>10.3f}   {desc[kind]}")

    t = pd.DataFrame(rows)
    t.to_csv(LOGS / "table4_context.csv", index=False)
    gain = t.loc[t.model == "payload_sink", "auc"].iloc[0] - t.loc[t.model == "payload_only", "auc"].iloc[0]
    print(f"\nvalue of knowing the output context: {gain:+.3f} AUC over payload alone")
    print("A request-side detector cannot observe the sink, so that gain is the part of")
    print("exploitability it is structurally unable to reach.")
    print(f"-> {LOGS / 'table4_context.csv'}")


if __name__ == "__main__":
    main()
