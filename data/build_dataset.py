"""Build 1,000 deduplicated originals (500 attack-like / 500 benign) from public sources.

Output: data/originals.csv and data/build_stats.json (counts removed at every step).
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, SEED, norm  # noqa: E402

RAW = DATA / "raw"
QUOTA = {  # (source, label) -> n
    ("kaggle", 1): 350, ("kaggle", 0): 300,
    ("payloadbox", 1): 150,
    ("mendeley", 0): 150,
    ("fixture", 0): 50,
}
LICENSE = {
    "kaggle": "see SOURCES.md (Kaggle dataset page)",
    "payloadbox": "MIT (SecLists mirror)",
    "mendeley": "CC BY 4.0",
    "fixture": "own work",
}
# Already-transformed look: percent-encoding, entities, \x escapes, long base64 runs
ENCODED = re.compile(r"%[0-9a-fA-F]{2}|&#|\\x[0-9a-fA-F]{2}|[A-Za-z0-9+/]{30,}={0,2}")
SYNTHETIC = re.compile(r"\b(?:LT|GT|SCHEME)_\d+")  # Mendeley placeholder tokens


# `norm` is imported from common.py: the same function normalises `norm_text` here and
# the live query string in the proxy, so training and serving see one representation.


def skeleton(s):
    s = norm(s)
    s = re.sub(r"(['\"]).*?\1", "S", s)  # string literals
    return re.sub(r"\d+", "0", s)        # numbers


def counts(df):
    return {f"{s}_{l}": int(n) for (s, l), n in df.groupby(["source", "label"]).size().items()}


def is_text(s):
    return s.isprintable() or all(c.isprintable() or c in "\t\n\r" for c in s)


def load_sources(stats):
    k = pd.read_csv(RAW / "kaggle" / "XSS_dataset.csv")
    kaggle = pd.DataFrame({"text": k.Sentence.astype(str), "label": k.Label.astype(int),
                           "source": "kaggle", "source_row_id": k["Unnamed: 0"].astype(str)})

    lines = (RAW / "seclists" / "XSS-payloadbox.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    pb = pd.DataFrame({"text": lines, "label": 1, "source": "payloadbox",
                       "source_row_id": [f"line{i + 1}" for i in range(len(lines))]})

    m = pd.read_csv(RAW / "mendeley" / "xss_dataset.csv")
    m = m[m.Label == 0]
    m = pd.DataFrame({"text": m.Query.astype(str), "label": 0, "source": "mendeley",
                      "source_row_id": m.index.astype(str)})
    before = len(m)
    m = m[~m.text.str.contains(ENCODED) & ~m.text.str.contains(SYNTHETIC)]
    stats["mendeley_benign_dropped_encoded_or_synthetic"] = before - len(m)

    fx = [l for l in (DATA / "fixtures_benign.txt").read_text().splitlines() if l.strip()]
    fixture = pd.DataFrame({"text": fx, "label": 0, "source": "fixture",
                            "source_row_id": [f"fx{i + 1}" for i in range(len(fx))]})
    # fixtures first so they survive cross-source dedup
    return pd.concat([fixture, kaggle, pb, m], ignore_index=True)


def main():
    stats = {}
    pool = load_sources(stats)
    stats["pool_raw"] = counts(pool)

    pool = pool[pool.text.map(is_text)]
    pool = pool[pool.text.str.strip().str.len().between(3, 500)]
    stats["after_length_and_text_filter"] = len(pool)

    pool["norm_text"] = pool.text.map(norm)
    conflict = pool.groupby("norm_text").label.nunique()
    conflict = set(conflict[conflict > 1].index)
    stats["dropped_label_conflicts"] = int(pool.norm_text.isin(conflict).sum())
    pool = pool[~pool.norm_text.isin(conflict)]
    n = len(pool)
    pool = pool.drop_duplicates("norm_text")
    stats["dropped_exact_duplicates"] = n - len(pool)

    # Keep one row per skeleton within a source, so a quota is not filled with near-copies
    pool["skeleton"] = pool.text.map(skeleton)
    n = len(pool)
    pool = pool.drop_duplicates(["source", "skeleton"])
    stats["dropped_within_source_near_duplicates"] = n - len(pool)
    stats["available_after_cleaning"] = counts(pool)

    parts = []
    for (src, lab), q in QUOTA.items():
        g = pool[(pool.source == src) & (pool.label == lab)]
        if len(g) < q:
            raise SystemExit(f"not enough rows for {src}/{lab}: {len(g)} < {q}")
        parts.append(g.sample(n=q, random_state=SEED))
    df = pd.concat(parts, ignore_index=True)

    # group AFTER pooling, so near-copies from different sources share a group
    df["group_id"] = pd.factorize(df.skeleton)[0]
    stats["n_originals"] = len(df)
    stats["n_groups"] = int(df.group_id.nunique())
    stats["multi_member_groups"] = int((df.group_id.value_counts() > 1).sum())

    df = df.rename(columns={"label": "label_attack_like"})
    df.insert(0, "id", [f"o{i:04d}" for i in range(len(df))])
    df["license"] = df.source.map(LICENSE)
    df["family"] = "original"
    df["parent_id"] = df.id
    df = df[["id", "text", "norm_text", "label_attack_like", "group_id", "source",
             "source_row_id", "license", "family", "parent_id"]]
    df.to_csv(DATA / "originals.csv", index=False)
    (DATA / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(df.groupby(["source", "label_attack_like"]).size())


if __name__ == "__main__":
    main()
