"""Published, externally-authored detectors, so the comparison is not our zoo vs itself.

Reviewer objection this answers: every detector evaluated so far was written by us, so
the finding could be a property of our modelling choices rather than of the task.

Two baselines nobody on this project designed:

  crs_regex   The XSS rules from the OWASP ModSecurity Core Rule Set
              (REQUEST-941-APPLICATION-ATTACK-XSS.conf, Apache-2.0), the ruleset deployed
              in front of a large share of the public web. We extract its @rx patterns and
              apply them directly. A real production detector, not a reimplementation.

  miczek_rf   The configuration the brief's own reference [1] reports: a random forest
              over character n-grams. Reconstructed from the paper's description, so it
              IS a reimplementation and is labelled as such.

Both are asked the same question we asked of our own models: does the verdict track
whether the payload ACTUALLY EXECUTES in a browser?

    PYTHONPATH=. python ml/published_baselines.py  ->  logs/table8_published.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, LOGS, SEED  # noqa: E402
from ml.split import split as gsplit  # noqa: E402

CRS = DATA / "raw" / "crs" / "REQUEST-941-APPLICATION-ATTACK-XSS.conf"


def load_crs_patterns():
    """Pull the @rx regexes out of the CRS config and compile what PCRE->Python allows."""
    text = CRS.read_text(encoding="utf-8", errors="replace")
    pats = re.findall(r'"@rx\s+(.*?)"\s*,?\s*\\?\s*\n', text, re.S)
    pats += re.findall(r'"@rx\s+([^"]+)"', text)
    out, bad = [], 0
    for p in dict.fromkeys(pats):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(re.compile(p, re.I))
        except re.error:
            bad += 1          # a few PCRE constructs have no Python equivalent
    return out, bad


class CRSDetector:
    """Fires if ANY CRS XSS pattern matches. No training; it is a fixed ruleset."""

    def __init__(self):
        self.pats, self.bad = load_crs_patterns()

    def score(self, texts):
        # fraction of rules matching: rankable for AUC, and >0 is the binary WAF decision
        n = max(len(self.pats), 1)
        return np.array([sum(bool(p.search(t)) for p in self.pats) / n for t in texts])

    def decide(self, texts):
        return np.array([int(any(p.search(t) for p in self.pats)) for t in texts])


def miczek_rf():
    """Char n-gram + random forest, the configuration reported in the brief's ref [1]."""
    return Pipeline([("v", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2)),
                     ("c", RandomForestClassifier(n_estimators=100, random_state=SEED,
                                                  n_jobs=-1))])


def main():
    m = pd.read_csv(DATA / "manifest.csv", keep_default_na=False)
    orig = m[m.label_kind == "original"].copy()
    cross = pd.read_csv(LOGS / "cross_context.csv", keep_default_na=False)
    cross = cross[cross.note == ""].copy()
    cross["ex"] = ((cross.fired_load == 1) | (cross.fired_interact == 1)).astype(int)

    crs = CRSDetector()
    print(f"CRS: {len(crs.pats)} usable @rx patterns ({crs.bad} unparseable in Python re)")

    rows = []
    for seed in (42, 1337, 2024):
        g2s = dict(zip(orig.group_id, gsplit(orig, seed).values))
        tro = m[(m.group_id.map(g2s) == "train") & (m.label_kind == "original")]
        sp = cross.group_id.map(g2s)
        te = cross[sp == "test"]
        y = te.ex.values
        txt = te.norm_text.tolist()

        s, d = crs.score(txt), crs.decide(txt)
        rows.append(dict(seed=seed, detector="crs_regex", trained_on="none (fixed ruleset)",
                         auc_exec=roc_auc_score(y, s), flag_rate=float(d.mean()),
                         recall_on_executing=float(d[y == 1].mean()),
                         flag_rate_on_inert=float(d[y == 0].mean()), n_test=len(te)))

        rf = miczek_rf().fit(tro.norm_text.tolist(), tro.label_attack_like.values)
        p = rf.predict_proba(txt)[:, 1]
        rows.append(dict(seed=seed, detector="miczek_rf", trained_on="attack_like",
                         auc_exec=roc_auc_score(y, p), flag_rate=float((p >= .5).mean()),
                         recall_on_executing=float((p[y == 1] >= .5).mean()),
                         flag_rate_on_inert=float((p[y == 0] >= .5).mean()), n_test=len(te)))

        trc = cross[sp.isin(["train", "val"])]
        rf2 = miczek_rf().fit(trc.norm_text.tolist(), trc.ex.values)
        p2 = rf2.predict_proba(txt)[:, 1]
        rows.append(dict(seed=seed, detector="miczek_rf", trained_on="execution",
                         auc_exec=roc_auc_score(y, p2), flag_rate=float((p2 >= .5).mean()),
                         recall_on_executing=float((p2[y == 1] >= .5).mean()),
                         flag_rate_on_inert=float((p2[y == 0] >= .5).mean()), n_test=len(te)))

    t = pd.DataFrame(rows)
    t.to_csv(LOGS / "table8_published.csv", index=False)

    print("\nAUC at predicting EXECUTION, mean +/- sd over 3 data splits")
    print("-" * 74)
    for (det, tr), g in t.groupby(["detector", "trained_on"], sort=False):
        print(f"  {det:<11} trained on {tr:<22} {g.auc_exec.mean():.3f} +/- {g.auc_exec.std():.3f}")
    print("\nbehaviour at the operating point")
    print("-" * 74)
    for (det, tr), g in t.groupby(["detector", "trained_on"], sort=False):
        print(f"  {det:<11} {tr:<22} flags {g.flag_rate.mean():.3f} of traffic | "
              f"catches {g.recall_on_executing.mean():.3f} executing | "
              f"{g.flag_rate_on_inert.mean():.3f} inert")
    rf = t[t.detector == "miczek_rf"]
    cost = (rf[rf.trained_on == "execution"].auc_exec.mean()
            - rf[rf.trained_on == "attack_like"].auc_exec.mean())
    print(f"\nlabel cost for the PUBLISHED configuration: {cost:+.3f} AUC")
    print(f"-> {LOGS / 'table8_published.csv'}")


if __name__ == "__main__":
    main()
