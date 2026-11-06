"""Tor OPEN — blend soft votingu z toru classic (LogReg+RF+MLP) z CatBoost i HistGradientBoosting.

  * wspólne cechy z solutions/features.py; modele klasyczne bez cech pochodnych, boosting na wszystkich,
  * uśrednienie prawdopodobieństw 5 modeli z równą wagą (voting z classic liczy się jako 3 modele),
  * reguła decyzyjna pod BA: argmax p_k / prior_k (prior z danych treningowych).
LightGBM pominięty celowo: na macOS wymaga systemowego libomp (brew), co psuje odtwarzalność.
"""

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from common.evaluation import CLASSES
from solutions import classic
from solutions.features import build_features

SEED = 42
MODELS = ["VotingClassifier", "LogisticRegression", "RandomForestClassifier", "MLPClassifier",
          "CatBoostClassifier", "HistGradientBoostingClassifier"]
SUMMARY = ("blend: soft voting LogReg+RF+MLP (x3) + CatBoost + HistGB; kursy (Shin), Elo, pi-ratings, forma, "
           "tabela, H2H, FIFA składów/drużyn, cechy Poissona; decyzja argmax p/prior pod BA")

__all__ = ["build_features", "fit_predict", "MODELS", "SUMMARY"]


def _proba(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    P = model.predict_proba(X_test)
    return P[:, [list(model.classes_).index(c) for c in CLASSES]]


def fit_predict(X_train, y_train, X_test):
    p_vote = classic.predict_proba(X_train, y_train, X_test)
    p_cat = _proba(CatBoostClassifier(iterations=600, learning_rate=0.03, depth=5, l2_leaf_reg=10,
                                      verbose=0, random_seed=SEED, thread_count=-1, allow_writing_files=False),
                   X_train, y_train, X_test)
    p_hgb = _proba(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                                  min_samples_leaf=80, l2_regularization=5, max_features=0.5,
                                                  random_state=SEED),
                   X_train, y_train, X_test)
    P = (3 * p_vote + p_cat + p_hgb) / 5
    return classic.decide(P, y_train)
