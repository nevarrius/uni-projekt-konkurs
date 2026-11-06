"""Tor OPEN — dowolne metody (gradient boosting, stacking, własne sieci, ensembling czegokolwiek).

Aktualnie: najlepszy wg CV jest model klasyczny (LogReg) — blendy z CatBoost/HGB/MLP wypadały gorzej
(patrz EXPERIMENTS.md). Funkcja `fit_predict_cb_blend` zostaje jako udokumentowana alternatywa.
"""

import numpy as np
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

from solutions import classic
from solutions.decision import split_meta

SEED = 42
MODELS = ["LogisticRegression"]
SUMMARY = ("LogReg(balanced, C=0.003) — jak classic (lepszy na CV niż CatBoost/HGB/blendy); "
           "kursy bez marży + ratingi FIFA wyjściowej 11 (wg formacji) + kontekst")

build_features = classic.build_features
fit_predict = classic.fit_predict

__all__ = ["build_features", "fit_predict", "MODELS", "SUMMARY"]


def fit_predict_cb_blend(X_train, y_train, X_test):
    """Średnia prawdop. CatBoost + LogReg (cv=0.4701 — gorzej niż sama LogReg 0.4726)."""
    X_train, _ = split_meta(X_train)
    X_test, _ = split_meta(X_test)
    y_train = np.asarray(y_train)
    cb = CatBoostClassifier(iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=10,
                            random_seed=SEED, verbose=0, thread_count=-1)
    cb.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    lr = classic.make_model()
    lr.fit(X_train, y_train)
    p_cb = cb.predict_proba(X_test)[:, [list(cb.classes_).index(c) for c in lr.classes_]]
    return lr.classes_[np.argmax(lr.predict_proba(X_test) + p_cb, axis=1)]
