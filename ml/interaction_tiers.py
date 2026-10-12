"""How plausible is the interaction each payload needs?

Our headline execution label counts a payload as executing if the marker fires either on
plain render or after the gated event is dispatched. A reviewer will reasonably object
that some of those events are things no real user does: onbeforecut, ondragleave,
onpaste. If the context effect rests on those, it is an artefact.

So split firings into tiers by how routinely a browsing user performs the action, and
report both the execution rate and the context gain per tier. The tiers are declared
here rather than derived, because that judgement is ours and should be inspectable.

    PYTHONPATH=. python ml/interaction_tiers.py  ->  logs/table11_interaction_tiers.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import LOGS, SEED  # noqa: E402
from ml.split import split as gsplit  # noqa: E402

# Things an ordinary visitor does without being coaxed.
PLAUSIBLE = {"click", "mouseover", "mouseenter", "mousemove", "mouseout", "mouseleave",
             "pointerover", "pointerenter", "pointermove", "pointerout", "pointerleave",
             "focus", "focusin", "dblclick", "mousedown", "mouseup", "pointerdown",
             "pointerup", "blur", "focusout", "scroll", "wheel"}
# Things that need deliberate, unusual behaviour, or an attacker-supplied lure.
IMPLAUSIBLE = {"paste", "beforepaste", "copy", "beforecopy", "cut", "beforecut",
               "drag", "dragstart", "dragend", "dragenter", "dragleave", "dragover",
               "drop", "contextmenu", "keydown", "keyup", "keypress",
               "beforeactivate", "beforedeactivate", "activate", "deactivate",
               "popstate", "pause", "volumechange"}


def event_of(t):
    m = re.match(r"on_event:on(\w+)", str(t) or "")
    return m.group(1).lower() if m else None


def tier_of(row):
    if row.fired_load == 1:
        return "1_no_interaction"
    if row.fired_interact != 1:
        return "5_never_fired"
    e = event_of(row.trigger_type)
    if e in PLAUSIBLE:
        return "2_plausible_interaction"
    if e in IMPLAUSIBLE:
        return "3_implausible_interaction"
    return "4_unclassified_event"


def pipe():
    return Pipeline([("v", TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                           sublinear_tf=True, min_df=2)),
                     ("c", LogisticRegression(max_iter=3000, random_state=SEED))])


def gain_for(c, positive_tiers, orig):
    """Context gain when only the given tiers count as 'executes', averaged over splits."""
    out = []
    for seed in (42, 1337, 2024):
        g2s = dict(zip(orig.group_id, gsplit(orig, seed).values))
        sp = c.group_id.map(g2s)
        tr, te = c[sp.isin(["train", "val"])], c[sp == "test"]
        ytr = tr.tier.isin(positive_tiers).astype(int).values
        yte = te.tier.isin(positive_tiers).astype(int).values
        if len(set(yte)) < 2 or len(set(ytr)) < 2:
            continue
        p = pipe().fit(tr.norm_text.tolist(), ytr).predict_proba(te.norm_text.tolist())[:, 1]
        sx = [f"[{s}] {t}" for s, t in zip(tr.sink, tr.norm_text)]
        tx = [f"[{s}] {t}" for s, t in zip(te.sink, te.norm_text)]
        s = pipe().fit(sx, ytr).predict_proba(tx)[:, 1]
        out.append((roc_auc_score(yte, p), roc_auc_score(yte, s), int(yte.sum())))
    if not out:
        return None
    a = np.array([[x[0], x[1], x[2]] for x in out])
    return a[:, 0].mean(), a[:, 1].mean(), a[:, 1].mean() - a[:, 0].mean(), a[:, 2].mean()


def main():
    c = pd.read_csv(LOGS / "cross_context.csv", keep_default_na=False)
    c = c[c.note == ""].copy()
    c["tier"] = c.apply(tier_of, axis=1)
    m = pd.read_csv(Path(LOGS).parent / "data" / "manifest.csv", keep_default_na=False)
    orig = m[m.label_kind == "original"].copy()

    nonesc = c[~c.sink.str.contains("escap")]
    print("TIER DISTRIBUTION  (non-escaped sinks, app1 crossed data)")
    print("-" * 66)
    d = nonesc.tier.value_counts().sort_index()
    for k, v in d.items():
        print(f"  {k:<30} {v:>6}  {v / len(nonesc):.3f}")
    print(f"  {'TOTAL':<30} {len(nonesc):>6}")

    print("\nWhich events drive the 'plausible' tier?")
    pl = nonesc[nonesc.tier == "2_plausible_interaction"]
    ev = pl.trigger_type.map(event_of).value_counts().head(8)
    print("  " + ", ".join(f"{k}:{v}" for k, v in ev.items()))

    print("\nCONTEXT GAIN under progressively stricter definitions of 'executes'")
    print("-" * 66)
    rows = []
    defs = [
        ("render or any interaction", ["1_no_interaction", "2_plausible_interaction",
                                       "3_implausible_interaction", "4_unclassified_event"]),
        ("render or PLAUSIBLE interaction", ["1_no_interaction", "2_plausible_interaction"]),
        ("plausible interaction only", ["2_plausible_interaction"]),
        ("render only (strictest)", ["1_no_interaction"]),
    ]
    for name, tiers in defs:
        r = gain_for(c, tiers, orig)
        if r is None:
            print(f"  {name:<34} (too few positives)")
            continue
        p, s, g, n = r
        rows.append(dict(definition=name, payload_auc=p, sink_auc=s, gain=g, mean_test_pos=n))
        print(f"  {name:<34} payload {p:.3f}  sink {s:.3f}  gain {g:+.3f}  pos {n:.0f}")

    pd.DataFrame(rows).to_csv(LOGS / "table11_interaction_tiers.csv", index=False)
    print(f"\n-> {LOGS / 'table11_interaction_tiers.csv'}")


if __name__ == "__main__":
    main()
