"""Neural detectors with the same fit / predict_proba interface as the sklearn models.

Two of them, for two different reasons:

  TransformerDetector  a pretrained model (CodeBERT by default). It has seen real code
                       during pretraining, so if it still cannot tell an executing payload
                       from an inert one, the limitation is not a shallow-feature problem.
  CharCNNDetector      a small character CNN trained from scratch. No pretraining, no
                       n-gram vocabulary, learns its own filters. Controls for the
                       possibility that the result is an artefact of fixed n-gram lookup.

Both are deliberately given the same input the other detectors get: the normalised,
once-decoded request value and nothing else. That is the point of the comparison.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin

SEED = 42


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TransformerDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name="microsoft/codebert-base", max_len=256, epochs=4,
                 batch_size=16, lr=2e-5, seed=SEED, verbose=True):
        self.model_name = model_name
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = _device()
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        self.model_ = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2).to(dev)

        X, y = list(X), np.asarray(y, dtype=np.int64)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr)
        n = len(X)
        steps = max(1, (n // self.batch_size) * self.epochs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps,
                                                    pct_start=0.1)
        rng = np.random.default_rng(self.seed)
        self.model_.train()
        step = 0
        for ep in range(self.epochs):
            order = rng.permutation(n)
            # keep the running loss ON the device: calling .item() every batch forces a
            # full GPU pipeline sync each step, which dominated the runtime on MPS
            tot = torch.zeros((), device=dev)
            seen = 0
            for i in range(0, n, self.batch_size):
                idx = order[i:i + self.batch_size]
                enc = self.tokenizer_([X[j] for j in idx], truncation=True,
                                      max_length=self.max_len, padding=True,
                                      return_tensors="pt").to(dev)
                labels = torch.tensor(y[idx], device=dev)
                out = self.model_(**enc, labels=labels)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
                if step < steps - 1:
                    sched.step()
                step += 1
                opt.zero_grad()
                tot += out.loss.detach() * len(idx)
                seen += len(idx)
            if self.verbose:                      # one sync per epoch, not per batch
                print(f"    epoch {ep + 1}/{self.epochs}  loss {tot.item() / max(seen, 1):.4f}",
                      flush=True)
        self.classes_ = np.array([0, 1])
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        dev = _device()
        self.model_.eval().to(dev)
        X = list(X)
        out = []
        for i in range(0, len(X), 64):
            enc = self.tokenizer_(X[i:i + 64], truncation=True, max_length=self.max_len,
                                  padding=True, return_tensors="pt").to(dev)
            out.append(torch.softmax(self.model_(**enc).logits, -1).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 2))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _CharCNN(nn.Module):
    def __init__(self, vocab, emb=48, ch=128, widths=(3, 4, 5, 7)):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, ch, w, padding=w // 2) for w in widths])
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(ch * len(widths), 2)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        feats = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(feats, dim=1)))


class CharCNNDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, max_len=256, epochs=12, batch_size=32, lr=1e-3, seed=SEED, verbose=True):
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.verbose = verbose

    def _encode(self, X):
        out = np.zeros((len(X), self.max_len), dtype=np.int64)
        for i, s in enumerate(X):
            for j, c in enumerate(str(s)[:self.max_len]):
                out[i, j] = self.vocab_.get(c, 1)
        return out

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X, y = list(X), np.asarray(y, dtype=np.int64)
        chars = sorted({c for s in X for c in str(s)})
        # 0 = padding, 1 = unseen character
        self.vocab_ = {c: i + 2 for i, c in enumerate(chars)}
        dev = _device()
        self.model_ = _CharCNN(len(self.vocab_) + 2).to(dev)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        enc = torch.tensor(self._encode(X))
        lab = torch.tensor(y)
        rng = np.random.default_rng(self.seed)
        self.model_.train()
        for ep in range(self.epochs):
            order = rng.permutation(len(X))
            tot = 0.0
            for i in range(0, len(X), self.batch_size):
                idx = order[i:i + self.batch_size]
                xb, yb = enc[idx].to(dev), lab[idx].to(dev)
                loss = lossf(self.model_(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            if self.verbose and (ep + 1) % 4 == 0:
                print(f"    epoch {ep + 1}/{self.epochs}  loss {tot / len(X):.4f}")
        self.classes_ = np.array([0, 1])
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        dev = _device()
        self.model_.eval().to(dev)
        enc = torch.tensor(self._encode(list(X)))
        out = []
        for i in range(0, len(enc), 256):
            out.append(torch.softmax(self.model_(enc[i:i + 256].to(dev)), -1).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 2))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
