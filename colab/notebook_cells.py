"""Colab notebook cells. Replace the cells in your notebook with these, in order.

Each block below marked  # ===== CELL n =====  is one Colab cell.

Two experiments:
  Experiment 1  fine-tune on ATTACK-LIKENESS (what the literature trains on), then ask
                whether the resulting score predicts EXECUTION. Adds bootstrap CIs and
                exports per-string scores so analysis can be redone without a GPU.
  Experiment 2  fine-tune on EXECUTION itself, on the crossed payload x sink data, with
                and without the sink token. This is the transformer version of
                ml/context_crossed.py. Each variant is TRAINED on the input format it is
                tested with; the earlier run prepended the sink at test time to a model
                that never saw it, which is why its number came out below chance.

Runtime on a T4: roughly 25 minutes total.
"""

# ===== CELL 1 =====  setup
CELL_1 = r'''
!pip -q install "transformers>=4.40" torch scikit-learn pandas

import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

SEED, EPOCHS, BATCH, MAX_LEN, LR = 42, 4, 16, 256, 2e-5
MODELS = ["microsoft/codebert-base", "distilroberta-base"]
# add "microsoft/graphcodebert-base" or "google/canine-s" here if you want more

torch.manual_seed(SEED); np.random.seed(SEED)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEV, torch.cuda.get_device_name(0) if DEV.type=="cuda" else "")
'''

# ===== CELL 2 =====  upload data
CELL_2 = r'''
import os
from google.colab import files

need = {"manifest.csv": None, "cross_context.csv": None}
for n in list(need):
    for c in (n, f"data/{n}", f"logs/{n}"):
        if os.path.exists(c): need[n] = c
if not all(need.values()):
    print("Upload data/manifest.csv AND logs/cross_context.csv")
    up = files.upload()
    for k in up:
        if "manifest" in k: need["manifest.csv"] = k
        if "cross_context" in k: need["cross_context.csv"] = k

m = pd.read_csv(need["manifest.csv"], keep_default_na=False)
m["label_attack_like"] = m.label_attack_like.astype(int)

cross = pd.read_csv(need["cross_context.csv"], keep_default_na=False)
cross = cross[cross.note == ""].copy()
cross["ex"] = ((cross.fired_load == 1) | (cross.fired_interact == 1)).astype(int)

print(f"manifest {len(m)} rows | cross_context {len(cross)} rows")
print("cross execution rate by sink:")
print(cross.groupby("sink").ex.mean().round(3).to_string())
'''

# ===== CELL 3 =====  shared helpers
CELL_3 = r'''
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def _hex(s): return "".join(f"\\x{ord(c):02x}" for c in s)

RENAMES = [("'__xss'+'_fired'", "'__probe'+'_flag'"),
           (_hex("__xss_fired"), _hex("__probe_flag")),
           (".__xss_f&#105;r&#101;d=", ".__probe_fl&#97;g="),
           ("&#119;&#105;nd&#111;&#119;", "s&#101;lf"),
           ("__xss_fired", "__probe_flag"), ("window", "self")]

def rename_marker(t):
    for a, b in RENAMES: t = t.replace(a, b)
    return t

def finetune(model_name, texts, labels, epochs=EPOCHS):
    """Same recipe for every model so the architecture is the only variable."""
    tok = AutoTokenizer.from_pretrained(model_name)
    net = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(DEV)
    y = np.asarray(labels); X = list(texts)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    total = max(1, (len(X)//BATCH)*epochs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total, pct_start=.1)
    rng = np.random.default_rng(SEED); net.train(); step = 0
    for ep in range(epochs):
        order = rng.permutation(len(X))
        run = torch.zeros((), device=DEV)       # keep loss on-device: syncing every batch
        for i in range(0, len(X), BATCH):       # is what made this crawl on Apple MPS
            idx = order[i:i+BATCH]
            enc = tok([X[j] for j in idx], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(DEV)
            out = net(**enc, labels=torch.tensor(y[idx], device=DEV))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if step < total-1: sch.step()
            step += 1; opt.zero_grad()
            run += out.loss.detach()*len(idx)
        print(f"    epoch {ep+1}/{epochs}  loss {run.item()/len(X):.4f}")
    return tok, net

@torch.no_grad()
def score(tok, net, texts, bs=64):
    net.eval(); out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(DEV)
        out.append(F.softmax(net(**enc).logits, -1)[:,1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)

def boot_auc(y, s, n=2000, seed=SEED):
    """Real bootstrap CI, replacing the analytic approximation used earlier."""
    rng = np.random.default_rng(seed); y, s = np.asarray(y), np.asarray(s); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2: out.append(roc_auc_score(y[i], s[i]))
    return (float(np.percentile(out,2.5)), float(np.percentile(out,97.5))) if out else (np.nan,)*2

def pick_threshold(p, y):
    """Max F1 on validation, ties to fewer false positives. Matches ml/train.py."""
    u = np.unique(p); c = np.unique((u[:-1]+u[1:])/2) if len(u)>1 else np.array([0.5])
    best = (-1.0, 0.5)
    for t in c:
        pr_ = (p>=t).astype(int)
        tp=((pr_==1)&(y==1)).sum(); fp=((pr_==1)&(y==0)).sum(); fn=((pr_==0)&(y==1)).sum()
        pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
        f1 = 2*pr*rc/(pr+rc) if pr+rc else 0
        if f1 > best[0]: best = (f1, float(t))
    return best[1]

def free():
    import gc; gc.collect(); torch.cuda.empty_cache()
print("helpers ready")
'''

# ===== CELL 4 =====  Experiment 1: attack-likeness model, tested on execution
CELL_4 = r'''
train = m[(m.split=="train") & (m.label_kind=="original")]
val   = m[(m.split=="val")   & (m.label_kind=="original")]
test  = m[m.split=="test"]
att = test[test.label_kind=="attack_variant"].copy()
att["ex"] = ((att.browser_confirmed.astype(str)=="1") |
             (att.browser_confirmed_with_interaction.astype(str)=="1")).astype(int)
att_d = att.drop_duplicates("norm_text")                 # distinct strings, not rows
ben_d = test[test.label_kind=="fpr_probe"].drop_duplicates("norm_text")
orig  = test[test.label_kind=="original"]
print(f"train {len(train)} | test attack {len(att_d)} "
      f"(exec {att_d.ex.sum()} / inert {(1-att_d.ex).sum()}) | benign {len(ben_d)}")

exp1 = []
for name in MODELS:
    print(f"\n=== EXP1 {name}")
    tok, net = finetune(name, train.norm_text.tolist(), train.label_attack_like.values)
    thr   = pick_threshold(score(tok,net,val.norm_text.tolist()), val.label_attack_like.values)
    s_att = score(tok, net, att_d.norm_text.tolist())
    s_ben = score(tok, net, ben_d.norm_text.tolist())
    s_ren = score(tok, net, [rename_marker(t) for t in att_d.norm_text])
    s_o   = score(tok, net, orig.norm_text.tolist())

    auc_task = roc_auc_score(np.r_[np.ones(len(s_att)), np.zeros(len(s_ben))], np.r_[s_att,s_ben])
    auc_exec = roc_auc_score(att_d.ex.values, s_att)
    lo, hi   = boot_auc(att_d.ex.values, s_att)
    rec_var  = float((s_att>=thr).mean()); rec_ren = float((s_ren>=thr).mean())

    r = dict(model=name, threshold=round(thr,4),
             auc_attack_vs_benign=round(auc_task,3),
             auc_execute_vs_inert=round(auc_exec,3),
             exec_ci_low=round(lo,3), exec_ci_high=round(hi,3),
             above_chance=bool(lo>0.5),
             recall_ordinary=round(float((s_o[orig.label_attack_like.values==1]>=thr).mean()),3),
             recall_variants=round(rec_var,3), recall_after_rename=round(rec_ren,3),
             shortcut_drop=round(rec_var-rec_ren,3),
             n_exec=int(att_d.ex.sum()), n_inert=int((1-att_d.ex).sum()))
    exp1.append(r); print("   ", r)

    tag = name.split("/")[-1]
    pd.DataFrame({"norm_text":att_d.norm_text.values, "ex":att_d.ex.values,
                  "score":s_att, "score_renamed":s_ren}).to_csv(f"scores_{tag}_attack.csv", index=False)
    pd.DataFrame({"norm_text":ben_d.norm_text.values, "score":s_ben}).to_csv(f"scores_{tag}_benign.csv", index=False)
    del net; free()

exp1 = pd.DataFrame(exp1); exp1.to_csv("table5_transformers.csv", index=False)
print("\n", exp1.to_string(index=False))
'''

# ===== CELL 5 =====  Experiment 2: execution model, payload only vs payload + sink
CELL_5 = r'''
ctr = cross[cross.split.isin(["train","val"])]
cte = cross[cross.split=="test"]
assert not (set(ctr.group_id) & set(cte.group_id)), "group leak"
print(f"crossed: train+val {len(ctr)}, test {len(cte)} "
      f"(exec {cte.ex.sum()} / inert {(1-cte.ex).sum()})")

def fmt(d, with_sink):
    return [f"[{s}] {t}" for s,t in zip(d.sink,d.norm_text)] if with_sink else d.norm_text.tolist()

exp2 = []
for name in MODELS:
    for with_sink in (False, True):
        lbl = "payload+sink" if with_sink else "payload_only"
        print(f"\n=== EXP2 {name} [{lbl}]")
        # trained on the SAME format it is tested with, which the earlier run did not do
        tok, net = finetune(name, fmt(ctr, with_sink), ctr.ex.values, epochs=3)
        s = score(tok, net, fmt(cte, with_sink))
        auc = roc_auc_score(cte.ex.values, s); lo, hi = boot_auc(cte.ex.values, s)
        ap  = average_precision_score(cte.ex.values, s)
        r = dict(model=name, input=lbl, auc=round(auc,3),
                 ci_low=round(lo,3), ci_high=round(hi,3), avg_precision=round(ap,3),
                 n_train=len(ctr), n_test=len(cte))
        exp2.append(r); print("   ", r)
        del net; free()

exp2 = pd.DataFrame(exp2); exp2.to_csv("table6_context_transformer.csv", index=False)
print("\n", exp2.to_string(index=False))
for name in MODELS:
    g = exp2[exp2.model==name]
    if len(g)==2:
        d = g[g.input=="payload+sink"].auc.iloc[0] - g[g.input=="payload_only"].auc.iloc[0]
        print(f"{name}: sink is worth {d:+.3f} AUC")
'''

# ===== CELL 6 =====  compare against the local models and download
CELL_6 = r'''
print("LOCAL models, AUC(executes vs inert), 632 distinct strings:")
print("  hand_gb 0.382 | hand_rf 0.388 | char_lr 0.439 | hybrid 0.453 | char_svm 0.467 | word_lr 0.473")
print("\nLOCAL char n-gram on crossed data (ml/context_crossed.py):")
print("  payload_only 0.814 | sink_only 0.850 | payload+sink 0.971 (avg prec 0.471 -> 0.852)")
print("\nTRANSFORMERS, this run:")
print(exp1[["model","auc_attack_vs_benign","auc_execute_vs_inert",
            "exec_ci_low","exec_ci_high","above_chance","shortcut_drop"]].to_string(index=False))
print()
print(exp2.to_string(index=False))

import glob
from google.colab import files
for f in ["table5_transformers.csv","table6_context_transformer.csv"]+sorted(glob.glob("scores_*.csv")):
    files.download(f)
'''

if __name__ == "__main__":
    for i, c in enumerate([CELL_1, CELL_2, CELL_3, CELL_4, CELL_5, CELL_6], 1):
        print(f"\n{'='*70}\n===== CELL {i} =====\n{'='*70}{c}")
