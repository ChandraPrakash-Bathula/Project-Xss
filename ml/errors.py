"""Step 3: dump every test-split error, then pick 10 to explain (draft causes).

Errors are collected under BOTH label definitions, and each row says which one:
  error_label = attack_like        FN: attack-like input scored below threshold
                                   FP: benign input scored at/above threshold
  error_label = browser_confirmed  FP: flagged, but the variant does NOT execute when
                                   rendered (no browser effect) -> request content alone
                                   could not know this

Draft cause (heuristic, to be confirmed by reading each row):
  unseen representation           FN on a transformed variant whose surface form the
                                  baseline never saw (entity / js / unusual tags)
  inconsistent decoding           FN on double URL-encoding: the proxy decodes once, the
                                  /reflect-decode endpoint decodes twice
  missing rendering context       flagged attack-like text that does not execute (rendered
                                  as text, or needs a user event that never happens)
  benign code mistaken for attack FP on benign text containing HTML/JS-looking characters

Selection rule for the 10 (stated so it is not cherry-picked): per cause, take distinct
parents in a seeded random order (seed 42), allocating 3/3/2/2 to the causes with the
largest pools first; at most one row per parent.

Evidence per row: the top n-grams pushing the score up, and (for FN) the strongest
attack n-grams that the parent original had but this input lost.

    python ml/errors.py [model]   -> logs/errors_<model>.csv, logs/errors_selected_<model>.csv
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, LOGS, MODELS, SEED  # noqa: E402

CAUSES = ["unseen representation", "inconsistent decoding",
          "missing rendering context", "benign code mistaken for attack"]


def explainer(pipe):
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    coef = clf.coef_[0]
    inv = np.empty(len(vec.vocabulary_), dtype=object)
    for g, i in vec.vocabulary_.items():
        inv[i] = g

    def top_push(text, k=4):
        x = vec.transform([text]).tocoo()
        contrib = sorted(zip(x.col, x.data * coef[x.col]), key=lambda t: -t[1])
        return [f"{inv[i]!r}({c:+.2f})" for i, c in contrib[:k] if c > 0]

    def lost(text, parent_text, k=4):
        a = set(vec.transform([parent_text]).indices) - set(vec.transform([text]).indices)
        best = sorted(a, key=lambda i: -coef[i])[:k]
        return [f"{inv[i]!r}({coef[i]:+.2f})" for i in best if coef[i] > 0]

    return top_push, lost


def draft_cause(r):
    if r.error_label == "browser_confirmed":
        return "missing rendering context"
    if r.error_type == "FP":
        return "benign code mistaken for attack"
    if r.subtype == "double_urlencode":
        return "inconsistent decoding"
    return "unseen representation"


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    b = joblib.load(MODELS / f"{model}.joblib")
    pipe, thr = b["pipeline"], b["threshold"]
    top_push, lost = explainer(pipe)

    m = pd.read_csv(DATA / "manifest.csv", keep_default_na=False)
    test = m[m.split == "test"].copy()
    test["score"] = pipe.predict_proba(test.norm_text.tolist())[:, 1]
    test["pred"] = (test.score >= thr).astype(int)
    parent_text = m[m.label_kind == "original"].set_index("id").norm_text

    y = test.label_attack_like.astype(int)
    e1 = test[test.pred != y].assign(
        error_label="attack_like",
        error_type=np.where(test.loc[test.pred != y, "pred"] == 1, "FP", "FN"))
    av = test[test.label_kind == "attack_variant"]
    e2 = av[(av.pred == 1) & (av.browser_confirmed == "0")].assign(
        error_label="browser_confirmed", error_type="FP")
    err = pd.concat([e1, e2], ignore_index=True)
    err["draft_cause"] = err.apply(draft_cause, axis=1)

    err["evidence_top_ngrams"] = err.norm_text.map(lambda t: " ".join(top_push(t)))
    # Reference for "what did the transformation remove?": the marker-plain sibling (same
    # parent, untransformed; the control row shows that form is still detected), not the
    # alert-form original. Parent-independent templates use their un-encoded counterpart.
    plain = m[m.subtype == "single_urlencode"].set_index("parent_id").norm_text
    template_ref = {"entity_in_attr_value": '"><svg onload=window.__xss_fired=true>',
                    "js_string_breakout": '";window.__xss_fired=true;//'}

    def reference(r):
        return template_ref.get(r.subtype) or plain.get(r.parent_id) or parent_text.get(r.parent_id, r.norm_text)

    err["reference_text"] = [reference(r) if r.label_kind == "attack_variant" else ""
                             for r in err.itertuples()]
    err["evidence_lost_vs_reference"] = [
        " ".join(lost(t, ref)) if et == "FN" and ref else ""
        for t, ref, et in zip(err.norm_text, err.reference_text, err.error_type)]

    cols = ["id", "error_label", "error_type", "draft_cause", "label_kind", "family", "subtype",
            "endpoint", "trigger_type", "score", "label_attack_like", "browser_confirmed",
            "browser_confirmed_with_interaction", "wire_q", "text", "norm_text", "parent_id",
            "evidence_top_ngrams", "reference_text", "evidence_lost_vs_reference"]
    err = err[cols].sort_values(["draft_cause", "score"])
    out_all = LOGS / f"errors_{model}.csv"
    err.to_csv(out_all, index=False)

    # ---- select 10 by the stated rule ----
    pools = {c: err[err.draft_cause == c] for c in CAUSES}
    order = sorted(CAUSES, key=lambda c: -len(pools[c]))
    quota = dict(zip(order, [3, 3, 2, 2]))
    picked = []
    for c in order:
        pool = pools[c].sample(frac=1, random_state=SEED)
        # prefer variety: one row per parent, then per subtype
        pool = pool.drop_duplicates("parent_id")
        pool = pd.concat([pool.drop_duplicates("subtype"), pool]).drop_duplicates("id")
        picked.append(pool.head(quota[c]))
    sel = pd.concat(picked, ignore_index=True)
    # if a cause has fewer distinct parents than its quota, fill the remaining slots from
    # the largest pools (still distinct parents), so there are always 10
    for c in order:
        if len(sel) >= 10:
            break
        extra = pools[c][~pools[c].parent_id.isin(sel.parent_id)].sample(frac=1, random_state=SEED)
        sel = pd.concat([sel, extra.drop_duplicates("parent_id").head(10 - len(sel))],
                        ignore_index=True)
    sel.insert(0, "n", range(1, len(sel) + 1))
    sel.to_csv(LOGS / f"errors_selected_{model}.csv", index=False)

    print(f"model={model}  threshold={thr:.4f}")
    print(f"all test errors: {len(err)}  -> {out_all}")
    print(err.groupby(["error_label", "error_type", "draft_cause"]).size().to_string())
    print(f"\n10 selected (rule: seeded, distinct parents, 3/3/2/2 by pool size):")
    for _, r in sel.iterrows():
        print(f"\n#{r.n} [{r.draft_cause}] {r.error_label} {r.error_type}  score={r.score:.3f}"
              f"  {r.family}/{r.subtype or '-'}  {r.endpoint or '-'}")
        print(f"   input : {r.norm_text[:95]}")
        print(f"   pushes: {r.evidence_top_ngrams}")
        if r.evidence_lost_vs_reference:
            print(f"   lost  : {r.evidence_lost_vs_reference}  (vs {r.reference_text[:50]})")


if __name__ == "__main__":
    main()
