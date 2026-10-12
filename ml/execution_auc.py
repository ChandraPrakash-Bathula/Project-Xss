"""Does the detector's score carry any information about whether a payload EXECUTES?

Two questions, deliberately separated:

  AUC(attack vs benign)   the task every XSS paper reports. Can the score separate
                          attack-like strings from harmless ones?
  AUC(executes vs inert)  among attack-like strings only, can the same score separate
                          the ones a browser actually runs from the ones it does not?

0.5 on the second means the score tells you nothing about real exploitability. Below 0.5
means it is mildly anti-correlated: the payloads it scores highest are, if anything, the
ones less likely to run.

Counting is over DISTINCT strings. Two transformations emit one constant string across
~92 rows each, and leaving those as rows would let a single string dominate the estimate.
"Executes" means the marker fired on plain render or after the gated user event; both were
observed in a real browser, and the detector can see neither.

    PYTHONPATH=. python ml/execution_auc.py [model ...]   -> logs/table3_execution_auc.csv
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, LOGS, MODELS, SEED  # noqa: E402

DEFAULT = ["baseline", "char_svm", "word_lr", "hand_rf", "hand_gb", "hybrid_lr"]


def boot_auc(y, s, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            out.append(roc_auc_score(y[i], s[i]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan, np.nan)


def main():
    models = sys.argv[1:] or DEFAULT
    m = pd.read_csv(DATA / "manifest.csv", keep_default_na=False)
    test = m[m.split == "test"]

    att = test[test.label_kind == "attack_variant"].copy()
    wi = att.browser_confirmed_with_interaction.astype(str)
    att["ex"] = ((att.browser_confirmed.astype(str) == "1") | (wi == "1")).astype(int)
    att = att.drop_duplicates("norm_text")
    ben = test[test.label_kind == "fpr_probe"].drop_duplicates("norm_text")

    print(f"TEST attack variants: {len(att)} distinct strings "
          f"({att.ex.sum()} execute, {(1 - att.ex).sum()} inert)")
    print(f"TEST benign probes:   {len(ben)} distinct strings\n")
    print(f"{'model':<12}{'AUC attack/benign':>19}{'AUC execute/inert':>19}  {'95% CI':>16}")
    print("-" * 70)

    rows = []
    for name in models:
        b = joblib.load(MODELS / f"{name}.joblib")
        pipe = b["pipeline"]
        s_att = pipe.predict_proba(att.norm_text.tolist())[:, 1]
        s_ben = pipe.predict_proba(ben.norm_text.tolist())[:, 1]
        y_task = np.r_[np.ones(len(s_att)), np.zeros(len(s_ben))]
        auc_task = roc_auc_score(y_task, np.r_[s_att, s_ben])
        auc_exec = roc_auc_score(att.ex.values, s_att)
        lo, hi = boot_auc(att.ex.values, s_att)
        rows.append(dict(model=name, arch=b.get("arch", "?"),
                         n_exec=int(att.ex.sum()), n_inert=int((1 - att.ex).sum()),
                         auc_attack_vs_benign=auc_task, auc_execute_vs_inert=auc_exec,
                         ci_low=lo, ci_high=hi))
        print(f"{name:<12}{auc_task:>19.3f}{auc_exec:>19.3f}  [{lo:.3f}, {hi:.3f}]")

    pd.DataFrame(rows).to_csv(LOGS / "table3_execution_auc.csv", index=False)
    print("\n0.5 means the score carries no information about whether the payload runs.")
    print(f"-> {LOGS / 'table3_execution_auc.csv'}")


if __name__ == "__main__":
    main()
