"""What is knowing the output context actually worth? Measured on crossed data.

ml/context.py ran this on manifest.csv, where every variant carries the one endpoint its
transformation targeted. Payload and sink were therefore nearly collinear, the sink was
recoverable from the payload, and the measured value of context came out at +0.002 AUC.
That number was an artefact of how the data was generated, not a fact about detection.

logs/cross_context.csv fixes it: every payload was sent to every one of the five contexts
and verified in each, so the two variables are crossed by construction. 55% of payloads
change outcome depending only on where the value lands.

Four models predicting EXECUTION, grouped split, no group across train and test:

  payload_only    the request value alone, which is all a proxy can see
  sink_only       the context alone, payload discarded
  payload_sink    both, sink prepended as a token
  payload_x_sink  both, with payload tokens namespaced per sink so the model can express
                  "this shape matters here but not there"

    PYTHONPATH=. python ml/context_crossed.py  ->  logs/table4_context_crossed.csv
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

SRC = LOGS / "cross_context.csv"
OUT = LOGS / "table4_context_crossed.csv"


def load():
    d = pd.read_csv(SRC, keep_default_na=False)
    d = d[d.note == ""]                                   # drop the few failed checks
    d["ex"] = ((d.fired_load == 1) | (d.fired_interact == 1)).astype(int)
    return d


def build(kind):
    if kind == "sink_only":
        v = TfidfVectorizer(analyzer="word", token_pattern=r"\S+")
    else:
        v = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
    return Pipeline([("v", v), ("c", LogisticRegression(max_iter=3000, random_state=SEED))])


def inputs(kind, d):
    if kind == "payload_only":
        return d.norm_text.tolist()
    if kind == "sink_only":
        return d.sink.tolist()
    if kind == "payload_sink":
        return [f"[{s}] {t}" for s, t in zip(d.sink, d.norm_text)]
    if kind == "payload_x_sink":
        return [f"[{s}] " + " ".join(f"{s}⁣{tok}" for tok in t.split()) + f" {t}"
                for s, t in zip(d.sink, d.norm_text)]
    raise ValueError(kind)


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


def main():
    d = load()
    tr = d[d.split.isin(["train", "val"])]
    te = d[d.split == "test"]
    assert not (set(tr.group_id) & set(te.group_id)), "group leak between train and test"

    varies = d.groupby("payload_id").ex.nunique()
    print(f"crossed observations: {len(d)}  ({d.payload_id.nunique()} payloads x {d.sink.nunique()} sinks)")
    print(f"train+val {len(tr)}, test {len(te)}  |  test executes {te.ex.sum()} / inert {(1 - te.ex).sum()}")
    print(f"payloads whose outcome depends on the sink: {(varies > 1).sum()} "
          f"of {len(varies)} ({(varies > 1).mean():.0%})\n")

    print("execution rate per context (all data):")
    for s, g in d.groupby("sink"):
        print(f"   {s:<30} {g.ex.mean():.3f}   ({g.ex.sum()}/{len(g)})")

    print(f"\n{'model':<16}{'AUC':>8}{'95% CI':>18}{'AvgPrec':>10}   sees")
    print("-" * 76)
    desc = {"payload_only": "request value only (proxy's view)",
            "sink_only": "output context only",
            "payload_sink": "value + sink",
            "payload_x_sink": "value + sink + interactions"}
    rows = []
    for kind in ["payload_only", "sink_only", "payload_sink", "payload_x_sink"]:
        pipe = build(kind).fit(inputs(kind, tr), tr.ex.values)
        s = pipe.predict_proba(inputs(kind, te))[:, 1]
        auc = roc_auc_score(te.ex.values, s)
        ap = average_precision_score(te.ex.values, s)
        lo, hi = boot(te.ex.values, s, te.group_id.values)
        rows.append(dict(model=kind, auc=auc, ci_low=lo, ci_high=hi, avg_precision=ap,
                         n_train=len(tr), n_test=len(te), sees=desc[kind]))
        print(f"{kind:<16}{auc:>8.3f}   [{lo:.3f}, {hi:.3f}]{ap:>10.3f}   {desc[kind]}")

    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False)
    po = t.loc[t.model == "payload_only", "auc"].iloc[0]
    ps = t.loc[t.model == "payload_sink", "auc"].iloc[0]
    px = t.loc[t.model == "payload_x_sink", "auc"].iloc[0]
    print(f"\nvalue of the output context: {ps - po:+.3f} AUC (plain), {px - po:+.3f} (with interactions)")
    print("A request-side detector cannot observe the sink, so this gap is the portion of")
    print("exploitability it is structurally unable to reach, however good the model gets.")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
