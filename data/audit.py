"""50-row manual label audit (25 attack-like / 25 benign), stratified by source.

This script draws the seeded sample and pre-computes *flags* that point at rows worth
reading closely. The flags are hints only: the `verdict` column is filled in by hand and
the reported error rate is the manual one.

    python data/audit.py            # draw the sample -> data/audit.csv (verdict blank)
    python data/audit.py --report   # summarise the filled-in verdicts
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, SEED  # noqa: E402

OUT = DATA / "audit.csv"

# A payload needs something that can execute: a tag with an event handler, a script
# element, or a javascript: URL. A string with none of these is suspicious as "attack".
EXECUTABLE = re.compile(r"<\s*script|\bon[a-z]+\s*=|javascript\s*:|<\s*svg|<\s*img[^>]*\bon",
                        re.I)
MARKUP = re.compile(r"<[a-z!/][^>]*>", re.I)


def flags(text, label):
    f = []
    if label == 1 and not EXECUTABLE.search(text):
        f.append("attack-without-executable-construct")
    if label == 0 and EXECUTABLE.search(text):
        f.append("benign-with-executable-construct")
    if label == 0 and MARKUP.search(text):
        f.append("benign-contains-markup")
    if len(text) < 8:
        f.append("very-short")
    return ";".join(f)


def draw():
    df = pd.read_csv(DATA / "originals.csv", keep_default_na=False)
    parts = []
    for lab in (1, 0):
        pool = df[df.label_attack_like == lab]
        # stratify by source, proportional, 25 per class
        take = (pool.source.value_counts(normalize=True) * 25).round().astype(int)
        for src, n in take.items():
            if n:
                parts.append(pool[pool.source == src].sample(n=min(n, len(pool[pool.source == src])),
                                                             random_state=SEED))
    s = pd.concat(parts).drop_duplicates("id")
    # top up to exactly 50 if rounding lost a row
    if len(s) < 50:
        s = pd.concat([s, df[~df.id.isin(s.id)].sample(n=50 - len(s), random_state=SEED)])
    s = s.sample(frac=1, random_state=SEED).head(50)

    out = s[["id", "source", "label_attack_like", "text"]].copy()
    out["flags"] = [flags(t, l) for t, l in zip(out.text, out.label_attack_like)]
    out["verdict"] = ""        # filled by hand: correct | wrong | unclear
    out["note"] = ""
    out.to_csv(OUT, index=False)
    print(f"drew {len(out)} rows -> {OUT}")
    print(out.groupby(["label_attack_like", "source"]).size().to_string())
    print(f"\nrows carrying a flag: {(out['flags'] != '').sum()}")
    print(out[out["flags"] != ""]["flags"].value_counts().to_string())


def report():
    a = pd.read_csv(OUT, keep_default_na=False)
    done = a[a.verdict != ""]
    if done.empty:
        raise SystemExit("no verdicts filled in yet")
    wrong = (done.verdict == "wrong").sum()
    unclear = (done.verdict == "unclear").sum()
    print(f"audited: {len(done)}/50")
    print(done.verdict.value_counts().to_string())
    print(f"\nlabel error rate: {wrong}/{len(done)} = {wrong / len(done):.1%}"
          f"   (plus {unclear} unclear)")
    print("\nby source:")
    print(done.groupby("source").verdict.value_counts().unstack(fill_value=0).to_string())
    if done.verdict.eq("wrong").any():
        print("\nrows judged wrong:")
        for _, r in done[done.verdict == "wrong"].iterrows():
            print(f"  {r.id} [{r.source}] label={r.label_attack_like}  {r.note}")
            print(f"    {r.text[:100]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else draw()
