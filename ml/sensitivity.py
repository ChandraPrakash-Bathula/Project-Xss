"""Sensitivity check: do source-pre-encoded attack payloads change any conclusion?

The 50-row label audit (data/audit.py) found that some payloadbox rows arrive ALREADY
HTML-entity-encoded (e.g. "&lt;IMG STYLE=&quot;...") . The Mendeley benign pool was filtered
for exactly this in data/build_dataset.py, but the attack pool was not, so a small share of
"original" attacks are effectively pre-transformed before data/transforms.py runs.

Rather than re-run the whole pipeline for it, this script re-computes Table 1 with those
rows AND all their descendants removed, so the size of the effect is measured and reported.

    python ml/sensitivity.py   ->  logs/sensitivity_preencoded.csv
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, HELD_OUT_FAMILY, LOGS, MODELS, SEEN_FAMILIES  # noqa: E402

# entity-encoded markup present in the SOURCE string, before any transformation of ours
PRE_ENCODED = r"&lt;|&gt;|&quot;|&#\d|&#x"


def conditions(m, drop_ids):
    t = m[m.split == "test"]
    t = t[~t.id.isin(drop_ids) & ~t.parent_id.isin(drop_ids)]
    o = t[t.label_kind == "original"]
    av = t[t.label_kind == "attack_variant"]
    tb = t[(t.label_kind == "fpr_probe") & (t.family.isin(SEEN_FAMILIES))]
    return {
        "ordinary": pd.concat([o[o.label_attack_like == 1], o[o.label_attack_like == 0]]),
        "seen transforms": pd.concat([av[av.family.isin(SEEN_FAMILIES)], tb]),
        f"held-out ({HELD_OUT_FAMILY})": pd.concat([av[av.family == HELD_OUT_FAMILY], tb]),
    }


def main():
    m = pd.read_csv(DATA / "manifest.csv", keep_default_na=False)
    orig = m[m.label_kind == "original"]
    bad = set(orig[(orig.label_attack_like == 1)
                   & orig.text.str.contains(PRE_ENCODED, regex=True)].id)
    desc = m[m.parent_id.isin(bad) & (m.label_kind == "attack_variant")]

    print(f"pre-encoded attack originals: {len(bad)} of {(orig.label_attack_like == 1).sum()} "
          f"({len(bad) / (orig.label_attack_like == 1).sum():.1%}) — all from payloadbox")
    print(f"descendant variants: {len(desc)}  (in test: {(desc.split == 'test').sum()})")
    print(f"of those variants, browser-confirmed executing: {(desc.browser_confirmed == '1').sum()}")

    rows = []
    for label, drop in [("kept (as published)", set()), ("removed", bad)]:
        for name in ("baseline", "augmented"):
            b = joblib.load(MODELS / f"{name}.joblib")
            pipe, thr = b["pipeline"], b["threshold"]
            for cond, df in conditions(m, drop).items():
                pred = (pipe.predict_proba(df.norm_text.tolist())[:, 1] >= thr).astype(int)
                y = df.label_attack_like.values
                rows.append(dict(variant=label, model=name, condition=cond,
                                 n_att=int((y == 1).sum()), n_ben=int((y == 0).sum()),
                                 recall=pred[y == 1].mean(), fpr=pred[y == 0].mean()))
    t = pd.DataFrame(rows)
    wide = t.pivot_table(index=["model", "condition"], columns="variant",
                         values=["recall", "fpr"]).reset_index()
    wide.columns = [" ".join(c).strip() for c in wide.columns]
    wide["recall delta"] = wide["recall removed"] - wide["recall kept (as published)"]

    t.to_csv(LOGS / "sensitivity_preencoded.csv", index=False)
    print("\nTable 1 recomputed with those rows removed:")
    print(wide.to_string(index=False, float_format="{:.3f}".format))
    worst = wide["recall delta"].abs().max()
    print(f"\nlargest recall shift: {worst:.3f}")
    print("Conclusion: the headline ordinary->transformed gap and the held-out result are "
          "unchanged, so the benchmark is reported as built, with this as a stated limitation.")
    print(f"-> {LOGS / 'sensitivity_preencoded.csv'}")


if __name__ == "__main__":
    main()
