"""Colab (T4): transformers on execution-grounded XSS labels. Multi-seed, early stopping.

Paste this entire file into ONE Colab cell. Runtime > Change runtime type > T4 GPU.
Upload when prompted:  data/manifest.csv  and  logs/cross_context.csv

WHAT CHANGED FROM THE PREVIOUS RUN, AND WHY
-------------------------------------------
The previous version trained for a fixed number of epochs with a single seed. Two
problems came out of that, and both are fixed here.

1. The payload-only arm of Experiment 2 was still improving when training stopped
   (final loss 0.401 and 0.414, both falling). Since that arm is the baseline the
   context gain is measured against, stopping early inflates the gain. Every run now
   trains until a validation metric stops improving, and any run that hits the epoch
   ceiling without triggering early stopping is flagged UNDERTRAINED in the output.

2. Every number came from seed 42. With one run there is no way to tell a real model
   difference from seed noise. Everything now runs over several seeds and reports
   mean and standard deviation.

Evidence that the effect is not purely an undertraining artefact: the local char n-gram
logistic regression, which does converge, gives payload_only 0.814 and payload+sink
0.971, a gain of +0.157. The transformers should be compared against that.

TWO EXPERIMENTS
  EXP1  train on ATTACK-LIKENESS, test whether the score predicts EXECUTION
  EXP2  train on EXECUTION over crossed payload x sink data, with and without the sink

Runtime on a T4: roughly 45 to 90 minutes depending on how early stopping fires.
"""
import copy
import glob
import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                "transformers>=4.40", "torch", "scikit-learn", "pandas"], check=False)

import numpy as np                                                          # noqa: E402
import pandas as pd                                                         # noqa: E402
import torch                                                                # noqa: E402
import torch.nn.functional as F                                             # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score          # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

# --------------------------------------------------------------------------- config
MODELS = ["microsoft/codebert-base", "distilroberta-base"]
SEEDS = [42, 1337, 2024]
BATCH, MAX_LEN, LR = 16, 256, 2e-5
MAX_EPOCHS_TASK, MAX_EPOCHS_EXEC = 20, 15
PATIENCE = 3                 # epochs without validation improvement before stopping
MIN_DELTA = 1e-4

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEV} "
      f"{torch.cuda.get_device_name(0) if DEV.type == 'cuda' else '(NO GPU: set Runtime > T4)'}")
curves = []                  # per-epoch log for every run, saved at the end


# ----------------------------------------------------------------------- load inputs
def find_inputs():
    need = {"manifest.csv": None, "cross_context.csv": None}
    for n in list(need):
        for cand in (n, f"data/{n}", f"logs/{n}"):
            if os.path.exists(cand):
                need[n] = cand
    if not all(need.values()):
        from google.colab import files
        print("Upload BOTH data/manifest.csv and logs/cross_context.csv")
        up = files.upload()
        for k in up:
            if "manifest" in k:
                need["manifest.csv"] = k
            if "cross_context" in k:
                need["cross_context.csv"] = k
    missing = [k for k, v in need.items() if not v]
    if missing:
        sys.exit(f"missing required file(s): {missing}")
    return need


paths = find_inputs()
m = pd.read_csv(paths["manifest.csv"], keep_default_na=False)
m["label_attack_like"] = m.label_attack_like.astype(int)
cross = pd.read_csv(paths["cross_context.csv"], keep_default_na=False)
cross = cross[cross.note == ""].copy()
cross["ex"] = ((cross.fired_load == 1) | (cross.fired_interact == 1)).astype(int)
print(f"manifest {len(m)} rows | cross_context {len(cross)} rows")


# -------------------------------------------------------------------------- helpers
def _hex(s):
    return "".join(f"\\x{ord(c):02x}" for c in s)


RENAMES = [("'__xss'+'_fired'", "'__probe'+'_flag'"),
           (_hex("__xss_fired"), _hex("__probe_flag")),
           (".__xss_f&#105;r&#101;d=", ".__probe_fl&#97;g="),
           ("&#119;&#105;nd&#111;&#119;", "s&#101;lf"),
           ("__xss_fired", "__probe_flag"), ("window", "self")]


def rename_marker(t):
    for a, b in RENAMES:
        t = t.replace(a, b)
    return t


@torch.no_grad()
def score(tok, net, texts, bs=128):
    net.eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(DEV)
        out.append(F.softmax(net(**enc).logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_early_stop(model_name, Xtr, ytr, Xva, yva, max_epochs, seed, tag):
    """Train until validation AUC stops improving; restore the best weights.

    Returns (tok, net, epochs_run, best_val_auc, stopped_early). stopped_early is False
    when the epoch ceiling was reached, which means the run may still be undertrained
    and its numbers should be treated with suspicion.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    net = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(DEV)
    Xtr, ytr = list(Xtr), np.asarray(ytr)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    rng = np.random.default_rng(seed)

    best_auc, best_state, bad, ran = -1.0, None, 0, 0
    for ep in range(1, max_epochs + 1):
        net.train()
        order = rng.permutation(len(Xtr))
        run = torch.zeros((), device=DEV)
        for i in range(0, len(Xtr), BATCH):
            idx = order[i:i + BATCH]
            enc = tok([Xtr[j] for j in idx], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(DEV)
            out = net(**enc, labels=torch.tensor(ytr[idx], device=DEV))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            run += out.loss.detach() * len(idx)
        tr_loss = run.item() / len(Xtr)
        va = score(tok, net, list(Xva))
        va_auc = roc_auc_score(yva, va) if len(np.unique(yva)) == 2 else float("nan")
        # Stop on validation LOSS, not AUC. On the attack-likeness task val AUC hits
        # 1.0000 by epoch 2 in ~92% of epochs, so an AUC criterion can never improve and
        # early stopping degenerates into a fixed epoch count. Loss keeps discriminating
        # long after AUC saturates.
        eps = 1e-7
        va_arr = np.clip(np.asarray(va, dtype=float), eps, 1 - eps)
        yv = np.asarray(yva, dtype=float)
        va_loss = float(-(yv * np.log(va_arr) + (1 - yv) * np.log(1 - va_arr)).mean())
        monitor = -va_loss                      # higher is better, same as AUC
        ran = ep
        curves.append(dict(run=tag, seed=seed, epoch=ep, train_loss=round(tr_loss, 4),
                           val_auc=round(float(va_auc), 4), val_loss=round(va_loss, 4)))
        print(f"      epoch {ep:>2}  train_loss {tr_loss:.4f}  val_loss {va_loss:.4f}  "
              f"val_auc {va_auc:.4f}", flush=True)

        if monitor > best_auc + MIN_DELTA:
            best_auc, bad = monitor, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"      early stop at epoch {ep} (best val_loss {-best_auc:.4f})")
                break

    if best_state is not None:
        net.load_state_dict(best_state)
        net.to(DEV)
    stopped_early = ran < max_epochs
    if not stopped_early:
        print(f"      WARNING: hit the {max_epochs}-epoch ceiling without early stopping; "
              f"this run may be UNDERTRAINED")
    return tok, net, ran, float(best_auc), stopped_early


def boot_auc(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            out.append(roc_auc_score(y[i], s[i]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan,) * 2


def pick_threshold(p, y):
    u = np.unique(p)
    cands = np.unique((u[:-1] + u[1:]) / 2) if len(u) > 1 else np.array([0.5])
    best = (-1.0, 0.5)
    for t in cands:
        q = (p >= t).astype(int)
        tp = ((q == 1) & (y == 1)).sum(); fp = ((q == 1) & (y == 0)).sum()
        fn = ((q == 0) & (y == 1)).sum()
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1]


def free(net):
    del net
    import gc
    gc.collect()
    if DEV.type == "cuda":
        torch.cuda.empty_cache()


# ============================== EXPERIMENT 1 ========================================
train = m[(m.split == "train") & (m.label_kind == "original")]
val = m[(m.split == "val") & (m.label_kind == "original")]
test = m[m.split == "test"]
att = test[test.label_kind == "attack_variant"].copy()
att["ex"] = ((att.browser_confirmed.astype(str) == "1") |
             (att.browser_confirmed_with_interaction.astype(str) == "1")).astype(int)
att_d = att.drop_duplicates("norm_text")
ben_d = test[test.label_kind == "fpr_probe"].drop_duplicates("norm_text")
print(f"\nEXP1: train {len(train)}, val {len(val)}, test attack {len(att_d)} "
      f"(exec {att_d.ex.sum()}/inert {(1 - att_d.ex).sum()}), benign {len(ben_d)}")
print("note: val is used both for early stopping and for the threshold; test is untouched")

e1 = []
for name in MODELS:
    for seed in SEEDS:
        print(f"\n=== EXP1 {name} seed={seed}")
        tok, net, eps, bva, early = train_early_stop(
            name, train.norm_text.tolist(), train.label_attack_like.values,
            val.norm_text.tolist(), val.label_attack_like.values,
            MAX_EPOCHS_TASK, seed, f"exp1|{name}")
        thr = pick_threshold(score(tok, net, val.norm_text.tolist()),
                             val.label_attack_like.values)
        s_att = score(tok, net, att_d.norm_text.tolist())
        s_ben = score(tok, net, ben_d.norm_text.tolist())
        s_ren = score(tok, net, [rename_marker(t) for t in att_d.norm_text])
        auc_task = roc_auc_score(np.r_[np.ones(len(s_att)), np.zeros(len(s_ben))],
                                 np.r_[s_att, s_ben])
        auc_exec = roc_auc_score(att_d.ex.values, s_att)
        lo, hi = boot_auc(att_d.ex.values, s_att)
        rv, rr = float((s_att >= thr).mean()), float((s_ren >= thr).mean())
        e1.append(dict(model=name, seed=seed, epochs=eps, stopped_early=early,
                       auc_attack_vs_benign=auc_task, auc_execute_vs_inert=auc_exec,
                       exec_ci_low=lo, exec_ci_high=hi, above_chance=bool(lo > 0.5),
                       recall_variants=rv, recall_after_rename=rr, shortcut_drop=rv - rr))
        print(f"    task_auc {auc_task:.3f} | exec_auc {auc_exec:.3f} [{lo:.3f},{hi:.3f}] "
              f"| shortcut_drop {rv - rr:+.3f} | epochs {eps}")
        if seed == SEEDS[0]:
            tag = name.split("/")[-1]
            pd.DataFrame({"norm_text": att_d.norm_text.values, "ex": att_d.ex.values,
                          "score": s_att, "score_renamed": s_ren}).to_csv(
                f"scores_{tag}_attack.csv", index=False)
        free(net)

e1 = pd.DataFrame(e1)
e1.to_csv("table5_transformers_perseed.csv", index=False)
e1agg = e1.groupby("model").agg(
    n_seeds=("seed", "size"), epochs_mean=("epochs", "mean"),
    all_early=("stopped_early", "all"),
    task_auc_mean=("auc_attack_vs_benign", "mean"), task_auc_std=("auc_attack_vs_benign", "std"),
    exec_auc_mean=("auc_execute_vs_inert", "mean"), exec_auc_std=("auc_execute_vs_inert", "std"),
    shortcut_mean=("shortcut_drop", "mean"), shortcut_std=("shortcut_drop", "std")).reset_index()
e1agg.to_csv("table5_transformers.csv", index=False)


# ============================== EXPERIMENT 2 ========================================
ctr = cross[cross.split == "train"]
cva = cross[cross.split == "val"]
cte = cross[cross.split == "test"]
assert not (set(ctr.group_id) & set(cte.group_id)), "group leak train/test"
assert not (set(cva.group_id) & set(cte.group_id)), "group leak val/test"
print(f"\nEXP2: train {len(ctr)}, val {len(cva)}, test {len(cte)} "
      f"(exec {cte.ex.sum()}/inert {(1 - cte.ex).sum()})")


def fmt(d, with_sink):
    return ([f"[{s}] {t}" for s, t in zip(d.sink, d.norm_text)]
            if with_sink else d.norm_text.tolist())


e2 = []
for name in MODELS:
    for with_sink in (False, True):
        lbl = "payload+sink" if with_sink else "payload_only"
        for seed in SEEDS:
            print(f"\n=== EXP2 {name} [{lbl}] seed={seed}")
            tok, net, eps, bva, early = train_early_stop(
                name, fmt(ctr, with_sink), ctr.ex.values,
                fmt(cva, with_sink), cva.ex.values,
                MAX_EPOCHS_EXEC, seed, f"exp2|{name}|{lbl}")
            s = score(tok, net, fmt(cte, with_sink))
            auc = roc_auc_score(cte.ex.values, s)
            lo, hi = boot_auc(cte.ex.values, s)
            e2.append(dict(model=name, input=lbl, seed=seed, epochs=eps,
                           stopped_early=early, auc=auc, ci_low=lo, ci_high=hi,
                           avg_precision=average_precision_score(cte.ex.values, s)))
            print(f"    auc {auc:.3f} [{lo:.3f},{hi:.3f}] | epochs {eps} | early {early}")
            free(net)

e2 = pd.DataFrame(e2)
e2.to_csv("table6_context_perseed.csv", index=False)
e2agg = e2.groupby(["model", "input"]).agg(
    n_seeds=("seed", "size"), epochs_mean=("epochs", "mean"),
    all_early=("stopped_early", "all"),
    auc_mean=("auc", "mean"), auc_std=("auc", "std"),
    ap_mean=("avg_precision", "mean"), ap_std=("avg_precision", "std")).reset_index()
e2agg.to_csv("table6_context_transformer.csv", index=False)
pd.DataFrame(curves).to_csv("learning_curves.csv", index=False)


# ================================= SUMMARY ==========================================
pd.set_option("display.width", 200)
print("\n" + "=" * 100)
print("EXPERIMENT 1  attack-likeness model, tested on execution   (mean +/- sd over seeds)")
print("=" * 100)
for _, r in e1agg.iterrows():
    flag = "" if r.all_early else "   <-- SOME RUNS HIT THE EPOCH CEILING"
    print(f"  {r.model:<26} task {r.task_auc_mean:.3f}+/-{r.task_auc_std:.3f}   "
          f"exec {r.exec_auc_mean:.3f}+/-{r.exec_auc_std:.3f}   "
          f"shortcut {r.shortcut_mean:+.3f}+/-{r.shortcut_std:.3f}   "
          f"epochs {r.epochs_mean:.1f}{flag}")
print("\n  LOCAL models, AUC(executes vs inert): hand_gb 0.382 | hand_rf 0.388 |")
print("  char_lr 0.439 | hybrid 0.453 | char_svm 0.467 | word_lr 0.473")

print("\n" + "=" * 100)
print("EXPERIMENT 2  execution model on crossed data: what is the output context worth?")
print("=" * 100)
print(e2agg.to_string(index=False, float_format="{:.3f}".format))
for name in MODELS:
    g = e2agg[e2agg.model == name]
    if len(g) == 2:
        p = g[g.input == "payload_only"]
        s = g[g.input == "payload+sink"]
        gain = s.auc_mean.iloc[0] - p.auc_mean.iloc[0]
        warn = "" if (p.all_early.iloc[0] and s.all_early.iloc[0]) else "  (ceiling hit: treat as upper bound)"
        print(f"  {name}: sink worth {gain:+.3f} AUC{warn}")
print("\n  LOCAL char n-gram, fully converged: payload_only 0.814 | sink_only 0.850 |")
print("  payload+sink 0.971  -> gain +0.157. Compare the transformer gains against this.")

print("\nREAD THE WARNINGS: any run flagged as hitting the epoch ceiling did not converge,")
print("and if that run is the payload_only arm then the reported context gain is inflated.")

try:
    from google.colab import files
    for f in (["table5_transformers.csv", "table5_transformers_perseed.csv",
               "table6_context_transformer.csv", "table6_context_perseed.csv",
               "learning_curves.csv"] + sorted(glob.glob("scores_*.csv"))):
        files.download(f)
except Exception:
    print("\nsaved all CSVs locally")
