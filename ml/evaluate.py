"""Table 1: detection metrics on the TEST split.

PRIMARY  (label = label_attack_like, full n):
    ordinary            original attacks  vs  original benign
    seen transforms     transport+entity attack variants  vs  transport+entity benign probes
    held-out family     js attack variants                vs  transport+entity benign probes
  js has no benign variant, so the transformed-benign pool is reused as negatives there
  (stated in the report). This is the headline table.

SECONDARY (label = browser_confirmed, smaller n):
    Of the attack variants that ACTUALLY execute in a browser, what fraction does the
    detector flag (recall)? Reported per family, with the much smaller n made explicit,
    because confirmed-executing positives are concentrated in synthetic vectors
    (transport n is tiny). A limitation, stated as such.

    python ml/evaluate.py                       # baseline
    python ml/evaluate.py baseline augmented    # both, side by side (after Step 7)
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, HELD_OUT_FAMILY, MODELS, SEED, SEEN_FAMILIES  # noqa: E402

rng = np.random.default_rng(SEED)


def metrics(pred, y):
    pred, y = np.asarray(pred, int), np.asarray(y, int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if p and r and not np.isnan(p + r) else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return dict(n_att=tp + fn, n_ben=fp + tn, precision=p, recall=r, f1=f1, fpr=fpr)


def boot_ci(pred, y, groups, stat, n=1000):
    """Bootstrap a statistic by resampling GROUPS (correlated variants move together)."""
    pred, y, groups = np.asarray(pred, int), np.asarray(y, int), np.asarray(groups)
    uniq = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n):
        idx = np.concatenate([by[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        v = stat(pred[idx], y[idx])
        if not np.isnan(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def recall_stat(pred, y):
    m = (y == 1); return float((pred[m] == 1).mean()) if m.any() else float("nan")


def fpr_stat(pred, y):
    m = (y == 0); return float((pred[m] == 1).mean()) if m.any() else float("nan")


def predict(model, df):
    b = joblib.load(MODELS / f"{model}.joblib")
    proba = b["pipeline"].predict_proba(df.norm_text.tolist())[:, 1]
    return (proba >= b["threshold"]).astype(int), b["threshold"]


def main():
    models = sys.argv[1:] or ["baseline"]
    m = pd.read_csv(DATA / "manifest.csv", keep_default_na=False)
    test = m[m.split == "test"].copy()

    orig = test[test.label_kind == "original"]
    avar = test[test.label_kind == "attack_variant"].copy()
    avar["bc"] = avar.browser_confirmed.astype(int)
    tben = test[(test.label_kind == "fpr_probe") & (test.family.isin(SEEN_FAMILIES))]

    obenign = orig[orig.label_attack_like == 0]
    # marker-plain CONTROL: single_urlencode decodes to exactly the marker-converted original,
    # i.e. "alert(...) swapped for the marker, no transformation". It separates the effect of
    # the marker swap from the effect of the transformation itself.
    control = avar[avar.subtype == "single_urlencode"]
    conditions = {
        "ordinary": pd.concat([orig[orig.label_attack_like == 1], obenign]),
        "marker-plain (control)": pd.concat([control, obenign]),
        "seen transforms": pd.concat([avar[avar.family.isin(SEEN_FAMILIES)], tben]),
        f"held-out ({HELD_OUT_FAMILY})": pd.concat([avar[avar.family == HELD_OUT_FAMILY], tben]),
    }

    print("=" * 78)
    print("TABLE 1 (PRIMARY) — label = label_attack_like, TEST split, full n")
    print("=" * 78)
    rows = []
    for model in models:
        for name, df in conditions.items():
            pred, thr = predict(model, df)
            mt = metrics(pred, df.label_attack_like)
            rlo, rhi = boot_ci(pred, df.label_attack_like, df.group_id, recall_stat)
            flo, fhi = boot_ci(pred, df.label_attack_like, df.group_id, fpr_stat)
            pos = df[df.label_attack_like == 1]
            # recall counting each distinct string once: synthetic templates repeat one string
            # ~92 times, which would otherwise dominate the headline number
            dd = df.assign(_p=pred).drop_duplicates("norm_text")
            dpos = dd[dd.label_attack_like == 1]
            rows.append(dict(model=model, condition=name, label="attack_like", **mt,
                             recall_ci=f"[{rlo:.2f},{rhi:.2f}]", fpr_ci=f"[{flo:.2f},{fhi:.2f}]",
                             uniq_att_texts=pos.norm_text.nunique(),
                             recall_unique=float(dpos._p.mean()) if len(dpos) else float("nan")))
    t1 = pd.DataFrame(rows)
    fmt = ["precision", "recall", "recall_unique", "f1", "fpr"]
    show = t1.assign(**{k: t1[k].map("{:.3f}".format) for k in fmt})
    print(show[["model", "condition", "n_att", "uniq_att_texts", "n_ben", "precision", "recall",
                "recall_ci", "recall_unique", "f1", "fpr", "fpr_ci"]].to_string(index=False))
    t1.to_csv(DATA.parent / "logs" / "table1_primary.csv", index=False)

    # Per-subtype recall, and the held-out family split by whether the literal marker token
    # survives. If augmentation only helps where '__xss_fired' is visible, the gain is a
    # marker-token shortcut, not generalisation to the unseen family.
    print("\nrecall by subtype (attack variants, TEST) — label = attack_like")
    sub_rows = []
    for model in models:
        pred, _ = predict(model, avar)
        a = avar.assign(_pred=pred, literal_marker=avar.norm_text.str.contains("__xss_fired", regex=False))
        for (fam, st), g in a.groupby(["family", "subtype"]):
            sub_rows.append(dict(model=model, family=fam, subtype=st, n=len(g),
                                 uniq=g.norm_text.nunique(), recall=g._pred.mean()))
        h = a[a.family == HELD_OUT_FAMILY]
        for lit, g in h.groupby("literal_marker"):
            sub_rows.append(dict(model=model, family=HELD_OUT_FAMILY,
                                 subtype=f"[held-out, literal marker {'present' if lit else 'absent'}]",
                                 n=len(g), uniq=g.norm_text.nunique(), recall=g._pred.mean()))
    ts = pd.DataFrame(sub_rows)
    if len(models) > 1:
        wide = ts.pivot_table(index=["family", "subtype", "n", "uniq"], columns="model",
                              values="recall").reset_index()
        print(wide.to_string(index=False, float_format="{:.3f}".format))
    else:
        print(ts.drop(columns="model").to_string(index=False, float_format="{:.3f}".format))
    ts.to_csv(DATA.parent / "logs" / "table1_by_subtype.csv", index=False)

    print("\n" + "=" * 78)
    print("TABLE 1 (SECONDARY) — label = browser_confirmed, attack variants only, TEST")
    print("of the variants that ACTUALLY execute, fraction the detector flags (recall)")
    print("=" * 78)
    # "Executes" = fires on plain render OR after the gated user event. Both were measured
    # in a real browser; the second needs a user action, which the detector cannot observe
    # either way. Restricting to on-load only leaves a handful of distinct strings and the
    # resulting recall is one string's result repeated, so both are reported.
    wi = avar.get("browser_confirmed_with_interaction", pd.Series("", index=avar.index)).astype(str)
    avar["ex_any"] = ((avar.bc == 1) | (wi == "1")).astype(int)

    rows2 = []
    for model in models:
        pred, thr = predict(model, avar)
        avar["_pred"] = pred
        for fam in [*SEEN_FAMILIES, HELD_OUT_FAMILY]:
            g = avar[avar.family == fam]
            # collapse to distinct strings so a template repeated N times counts once
            gd = g.drop_duplicates("norm_text")
            ex, inert = gd[gd.ex_any == 1], gd[gd.ex_any == 0]
            onload = gd[gd.bc == 1]
            rows2.append(dict(
                model=model, family=fam,
                n_exec_distinct=len(ex),
                recall_on_executing=(ex._pred.mean() if len(ex) else float("nan")),
                n_onload_distinct=len(onload),
                recall_onload_only=(onload._pred.mean() if len(onload) else float("nan")),
                n_inert_distinct=len(inert),
                flag_rate_on_inert=(inert._pred.mean() if len(inert) else float("nan"))))
    t2 = pd.DataFrame(rows2)
    fmt2 = ["recall_on_executing", "recall_onload_only", "flag_rate_on_inert"]
    show2 = t2.assign(**{k: t2[k].map("{:.3f}".format) for k in fmt2})
    print(show2.to_string(index=False))
    t2.to_csv(DATA.parent / "logs" / "table1_secondary.csv", index=False)
    print("\nAll counts are DISTINCT STRINGS, not rows: entity_in_attr_value and")
    print("js_string_breakout are parent-independent templates that fill ~92 rows with one")
    print("string each, which would otherwise dominate the averages.")

    # ---- marker-token shortcut probe (detector-only; nothing is executed) ----
    # Every augmented training row contains the marker identifiers, and no benign row does.
    # Rename them consistently in each test variant (keeping structure and obfuscation) and
    # re-score. A model that learned attack STRUCTURE should keep its recall; one that
    # learned the marker NAME will lose it.
    print("\n" + "=" * 78)
    print("SHORTCUT PROBE — test attack variants with marker identifiers renamed")
    print("(window -> self, __xss_fired -> __probe_flag, incl. hex/entity/concat forms)")
    print("=" * 78)
    probe = avar.assign(norm_text=avar.norm_text.map(rename_marker))
    changed = (probe.norm_text != avar.norm_text).mean()
    rows3 = []
    for model in models:
        p_orig, _ = predict(model, avar)
        p_ren, _ = predict(model, probe)
        for fam in [*SEEN_FAMILIES, HELD_OUT_FAMILY]:
            k = (avar.family == fam).values
            rows3.append(dict(model=model, family=fam, n=int(k.sum()),
                              recall_original=p_orig[k].mean(), recall_renamed=p_ren[k].mean()))
    t3 = pd.DataFrame(rows3)
    t3["drop"] = t3.recall_original - t3.recall_renamed
    print(f"rows changed by rename: {changed:.0%}")
    print(t3.to_string(index=False, float_format="{:.3f}".format))
    t3.to_csv(DATA.parent / "logs" / "table1_shortcut_probe.csv", index=False)


def _hex(s):
    return "".join(f"\\x{ord(c):02x}" for c in s)


_RENAMES = [  # applied to the normalised (lower-cased) text
    ("'__xss'+'_fired'", "'__probe'+'_flag'"),
    (_hex("__xss_fired"), _hex("__probe_flag")),
    (".__xss_f&#105;r&#101;d=", ".__probe_fl&#97;g="),       # entity template
    ("&#119;&#105;nd&#111;&#119;", "s&#101;lf"),               # entity-encoded 'window'
    ("__xss_fired", "__probe_flag"),
    ("window", "self"),
]


def rename_marker(t):
    for a, b in _RENAMES:
        t = t.replace(a, b)
    return t


if __name__ == "__main__":
    main()
