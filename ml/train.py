"""Train the detector and freeze its decision threshold.

Baseline:   trained on ORIGINAL train rows only.
Augmented:  same pipeline, same hyper-parameters, plus verified variants of the SEEN
            families added to the train split only (Step 7). Only the data changes.

    python ml/train.py                 # -> models/baseline.joblib
    python ml/train.py --augmented     # -> models/augmented.joblib  (after Step 5)

The saved bundle carries the pipeline, the threshold and the provenance of both, so the
proxy and evaluate.py cannot drift from what was fitted here.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, HELD_OUT_FAMILY, MODELS, SEED, SEEN_FAMILIES  # noqa: E402

# Fixed for BOTH models. Changing anything here invalidates the baseline/augmented comparison.
VECTORIZER = dict(analyzer="char", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
CLASSIFIER = dict(C=1.0, max_iter=1000, random_state=SEED)


def build_pipeline(arch="char_lr"):
    """char_lr is the project baseline, kept inline so it never silently changes.

    Any other architecture comes from ml/zoo.py. Everything else about training
    (splits, threshold rule, leak guards, bundle format) is identical across them,
    so the architecture is the only variable.
    """
    if arch == "char_lr":
        return Pipeline([
            ("tfidf", TfidfVectorizer(**VECTORIZER)),  # fitted in-pipeline => train only
            ("clf", LogisticRegression(**CLASSIFIER)),
        ])
    from ml.zoo import build as build_arch
    return build_arch(arch)


def load_manifest():
    """manifest.csv once variants exist (Step 5), originals.csv before that."""
    path = DATA / "manifest.csv"
    if not path.exists():
        path = DATA / "originals.csv"
    df = pd.read_csv(path, keep_default_na=False)
    df["label_attack_like"] = df.label_attack_like.astype(int)
    if "family" not in df:
        df["family"] = "original"
    return df, path


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"n": len(y_true), "n_attack": tp + fn, "n_benign": fp + tn, "tp": tp,
            "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "f1": f1, "fpr": fpr}


def pick_threshold(pipe, X_val, y_val):
    """Threshold that maximises F1 on validation. Ties break toward the lower FPR."""
    proba = pipe.predict_proba(X_val)[:, 1]
    # candidates: midpoints between observed scores, so every distinct split is tried
    uniq = np.unique(proba)
    cands = np.unique(np.concatenate([[0.5], (uniq[:-1] + uniq[1:]) / 2])) if len(uniq) > 1 else np.array([0.5])
    best = None
    for t in cands:
        m = metrics(y_val, (proba >= t).astype(int))
        key = (m["f1"], -m["fpr"], -abs(t - 0.5))
        if best is None or key > best[0]:
            best = (key, float(t), m)
    return best[1], best[2]


def report(name, m):
    print(f"  {name:<22} n={m['n']:<5d} (att {m['n_attack']}/ben {m['n_benign']})  "
          f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
          f"FPR={m['fpr']:.3f}   [tp {m['tp']} fp {m['fp']} fn {m['fn']} tn {m['tn']}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augmented", action="store_true",
                    help="add verified SEEN-family variants to the training split (Step 7)")
    ap.add_argument("--arch", default="char_lr",
                    help="detector architecture from ml/zoo.py (default: the char_lr baseline)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df, path = load_manifest()
    stem = "baseline" if args.arch == "char_lr" else args.arch
    name = f"{stem}_augmented" if args.augmented else stem
    print(f"manifest: {path.name}  rows={len(df)}  model={name}")

    train = df[(df.split == "train") & (df.family == "original")]
    val = df[(df.split == "val") & (df.family == "original")]
    aug_stats = {}
    if args.augmented:
        kind = df.get("label_kind", pd.Series("", index=df.index))
        extra = df[(df.split == "train") & (kind == "attack_variant")
                   & (df.family.isin(SEEN_FAMILIES))]
        # "verified executing" = fired from plain rendering OR after the gated user event
        on_load = extra.browser_confirmed.astype(str) == "1"
        on_act = extra.get("browser_confirmed_with_interaction",
                           pd.Series("", index=extra.index)).astype(str) == "1"
        extra = extra[on_load | on_act]
        aug_stats["verified_candidates"] = len(extra)
        # exact-text leak guard: synthetic templates are parent-independent, so the same
        # string can sit in val/test under another group. Never train on such a string.
        held = set(df[df.split.isin(["val", "test"])].norm_text)
        leak = extra.norm_text.isin(held)
        aug_stats["dropped_text_in_val_or_test"] = int(leak.sum())
        extra = extra[~leak]
        # drop duplicates of each other and of the originals already in train
        n = len(extra)
        extra = extra.drop_duplicates("norm_text")
        extra = extra[~extra.norm_text.isin(set(train.norm_text))]
        aug_stats["dropped_duplicates"] = n - len(extra)
        if extra.empty:
            raise SystemExit("no usable verified SEEN-family variants in the train split")
        aug_stats["added"] = len(extra)
        aug_stats["added_by_subtype"] = extra.subtype.value_counts().to_dict()
        train = pd.concat([train, extra], ignore_index=True)
        print(f"  + {len(extra)} verified variants from {SEEN_FAMILIES} "
              f"(held-out family '{HELD_OUT_FAMILY}' excluded from train AND val)")
        print(f"    {aug_stats}")

    # Guard rails: the comparison is only meaningful if these hold.
    assert set(train.group_id) & set(val.group_id) == set(), "group leak train/val"
    assert HELD_OUT_FAMILY not in set(train.family), "held-out family leaked into train"
    assert HELD_OUT_FAMILY not in set(val.family), "held-out family leaked into val"
    test_text = set(df[df.split == "test"].norm_text)
    assert not (set(train.norm_text) & test_text), "exact train text appears in test"

    Xtr, ytr = train.norm_text.tolist(), train.label_attack_like.tolist()
    Xva, yva = val.norm_text.tolist(), val.label_attack_like.tolist()

    pipe = build_pipeline(args.arch).fit(Xtr, ytr)
    thresh, val_m = pick_threshold(pipe, Xva, yva)
    train_m = metrics(ytr, (pipe.predict_proba(Xtr)[:, 1] >= thresh).astype(int))

    print(f"\nthreshold (max F1 on validation) = {thresh:.4f}")
    report("train", train_m)
    report("validation", val_m)
    if hasattr(pipe, "named_steps") and "tfidf" in pipe.named_steps:
        print(f"  vocabulary: {len(pipe.named_steps['tfidf'].vocabulary_):,} n-grams")

    MODELS.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else MODELS / f"{name}.joblib"
    joblib.dump({
        "pipeline": pipe,
        "threshold": thresh,
        "name": name,
        "arch": args.arch,
        "input_form": "common.norm(decode_once(query_string))",
        "vectorizer_params": VECTORIZER,
        "classifier_params": CLASSIFIER,
        "seed": SEED,
        "manifest": path.name,
        "n_train": len(train),
        "train_families": sorted(set(train.family)),
        "held_out_family": HELD_OUT_FAMILY,
        "augmentation": aug_stats,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": {"train": train_m, "validation": val_m},
    }, out)
    print(f"saved -> {out}")

    MODELS.joinpath(f"{name}_trainlog.json").write_text(json.dumps(
        {"threshold": thresh, "train": train_m, "validation": val_m,
         "n_train": len(train), "families": sorted(set(train.family))}, indent=2))


if __name__ == "__main__":
    main()
