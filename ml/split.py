"""Stratified group split of originals: folds 0-2 train, 3 validation, 4 test.

Variants inherit their parent's split later (make_manifest.py), so a group never crosses splits.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, SEED  # noqa: E402


def assert_no_group_leak(df):
    per_group = df.groupby("group_id").split.nunique()
    leaked = per_group[per_group > 1]
    assert leaked.empty, f"group leakage across splits: {list(leaked.index)[:10]}"


def split(df, seed=SEED):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    names = {0: "train", 1: "train", 2: "train", 3: "val", 4: "test"}
    out = pd.Series(index=df.index, dtype=object)
    for k, (_, idx) in enumerate(sgkf.split(df, df.label_attack_like, groups=df.group_id)):
        out.iloc[idx] = names[k]
    return out


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else SEED
    df = pd.read_csv(DATA / "originals.csv", keep_default_na=False)
    df["split"] = split(df, seed)
    assert_no_group_leak(df)
    df.to_csv(DATA / "originals.csv", index=False)
    df[["id", "group_id", "split"]].to_csv(DATA / f"split_ids_seed{seed}.csv", index=False)
    print(df.groupby(["split", "label_attack_like"]).size().unstack())
    print(pd.crosstab(df.source, df.split))


if __name__ == "__main__":
    main()
