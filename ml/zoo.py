"""Detector architectures, all exposing the same sklearn-style predict_proba interface.

The point of the zoo is NOT to find a better detector. It is to test whether the
findings from the char n-gram baseline are a property of that one model or a property
of request-content XSS detection in general. If the inversion (flagging inert payloads
while missing executing ones) holds across every architecture, it is a statement about
the task. If it only holds for one, there is no general claim to make.

Every architecture is trained on the same splits, with the same threshold-selection
rule, and evaluated by the same code, so the only thing that varies is the model.

    char_lr     char n-gram TF-IDF + logistic regression   (the project baseline)
    char_svm    char n-gram TF-IDF + calibrated linear SVM
    word_lr     word/token TF-IDF + logistic regression     (tokenisation ablation)
    hand_rf     hand-crafted structural features + random forest
    hand_gb     the same features + gradient boosting
    hybrid_lr   char n-grams + hand-crafted features, union
"""
import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

SEED = 42

# Tokeniser that keeps punctuation as tokens. A plain word tokeniser throws away the
# characters that matter most here (< > = " ' / ;), which is itself worth showing.
_WORD_RE = r"[A-Za-z_][A-Za-z0-9_]*|[<>=\"'/;:(){}\[\]&#%\\?.,+-]"

# Event handlers and sinks a human analyst would look for. Deliberately the kind of
# feature set a rule-driven engineer would hand-write.
_HANDLERS = re.compile(r"\bon[a-z]{3,20}\s*=", re.I)
_SCRIPTISH = re.compile(r"<\s*script|javascript\s*:|data\s*:text/html|srcdoc\s*=", re.I)
_SINKS = re.compile(r"\b(eval|settimeout|setinterval|function|document\.write|innerhtml|"
                    r"outerhtml|insertadjacenthtml|location|cookie|fetch|xmlhttprequest)\b", re.I)
_TAGS = re.compile(r"<\s*/?\s*([a-z][a-z0-9]*)", re.I)
_ENTITY = re.compile(r"&#x?[0-9a-f]+;|&[a-z]{2,10};", re.I)
_PCT = re.compile(r"%[0-9a-f]{2}", re.I)
_HEXESC = re.compile(r"\\x[0-9a-f]{2}|\\u[0-9a-f]{4}", re.I)


def _entropy(s):
    if not s:
        return 0.0
    _, counts = np.unique(np.frombuffer(s.encode("utf-8", "replace"), dtype=np.uint8),
                          return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """Structural features an analyst would write by hand, no learned vocabulary.

    This is the fairest non-n-gram comparison: it cannot memorise `alert`, because
    nothing here is a literal string lookup of a payload token. If this model shows the
    same inversion, the problem is not n-gram memorisation.
    """

    NAMES = ["len", "entropy", "n_lt", "n_gt", "n_quote", "n_squote", "n_eq", "n_paren",
             "n_semi", "n_slash", "n_amp", "n_pct", "n_backslash", "n_brace", "n_bracket",
             "has_handler", "n_handlers", "has_scriptish", "n_sinks", "n_tags",
             "n_entities", "n_pcths", "n_hexesc", "punct_ratio", "digit_ratio",
             "upper_ratio", "nonascii_ratio", "max_token_len", "ws_ratio", "balanced_lt_gt"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for s in X:
            s = str(s)
            n = max(len(s), 1)
            lt, gt = s.count("<"), s.count(">")
            punct = sum(not c.isalnum() and not c.isspace() for c in s)
            toks = re.findall(r"\S+", s)
            rows.append([
                len(s), _entropy(s), lt, gt, s.count('"'), s.count("'"), s.count("="),
                s.count("(") + s.count(")"), s.count(";"), s.count("/"), s.count("&"),
                s.count("%"), s.count("\\"), s.count("{") + s.count("}"),
                s.count("[") + s.count("]"),
                1.0 if _HANDLERS.search(s) else 0.0, len(_HANDLERS.findall(s)),
                1.0 if _SCRIPTISH.search(s) else 0.0, len(_SINKS.findall(s)),
                len(_TAGS.findall(s)), len(_ENTITY.findall(s)), len(_PCT.findall(s)),
                len(_HEXESC.findall(s)), punct / n,
                sum(c.isdigit() for c in s) / n, sum(c.isupper() for c in s) / n,
                sum(ord(c) > 127 for c in s) / n,
                max((len(t) for t in toks), default=0),
                sum(c.isspace() for c in s) / n, 1.0 if lt == gt else 0.0,
            ])
        return np.asarray(rows, dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.NAMES, dtype=object)


class SparseHandcrafted(BaseEstimator, TransformerMixin):
    """Hand-crafted features, scaled, returned sparse so they can sit beside TF-IDF.

    Done as one transformer rather than a nested Pipeline because a Pipeline inside a
    FeatureUnion is not refitted correctly in this sklearn version.
    """

    def __init__(self):
        self.feat = HandcraftedFeatures()
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(self.feat.transform(X))
        return self

    def transform(self, X):
        return csr_matrix(self.scaler.transform(self.feat.transform(X)))


def _char_tfidf():
    return TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True, min_df=2)


def build(arch):
    """Return an unfitted pipeline for the named architecture."""
    if arch == "char_lr":
        return Pipeline([("tfidf", _char_tfidf()),
                         ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))])

    if arch == "char_svm":
        # LinearSVC has no predict_proba, so calibrate it to keep the interface identical
        return Pipeline([("tfidf", _char_tfidf()),
                         ("clf", CalibratedClassifierCV(
                             LinearSVC(C=1.0, random_state=SEED, dual="auto"),
                             cv=3, method="sigmoid"))])

    if arch == "word_lr":
        return Pipeline([("tfidf", TfidfVectorizer(analyzer="word", token_pattern=_WORD_RE,
                                                   ngram_range=(1, 2), sublinear_tf=True,
                                                   min_df=2)),
                         ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))])

    if arch == "hand_rf":
        return Pipeline([("feat", HandcraftedFeatures()), ("scale", StandardScaler()),
                         ("clf", RandomForestClassifier(n_estimators=400, random_state=SEED,
                                                        n_jobs=-1, min_samples_leaf=2))])

    if arch == "hand_gb":
        return Pipeline([("feat", HandcraftedFeatures()), ("scale", StandardScaler()),
                         ("clf", GradientBoostingClassifier(random_state=SEED))])

    if arch == "hybrid_lr":
        return Pipeline([
            ("u", FeatureUnion([("char", _char_tfidf()), ("hand", SparseHandcrafted())])),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))])

    if arch in ("codebert", "charcnn"):
        # imported lazily so the sklearn models never pay the torch import cost
        from ml.neural import CharCNNDetector, TransformerDetector
        if arch == "charcnn":
            return CharCNNDetector()
        return TransformerDetector(model_name="microsoft/codebert-base")

    raise ValueError(f"unknown architecture: {arch}")


ARCHES = ["char_lr", "char_svm", "word_lr", "hand_rf", "hand_gb", "hybrid_lr",
          "charcnn", "codebert"]
