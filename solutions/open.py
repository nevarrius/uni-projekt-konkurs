"""Tor OPEN — dowolne metody.

Architektura: solutions/stack.py; destylacja kursów = średnia (LightGBM ×3 regresory, LogReg soft).
"""

import numpy as np
from lightgbm import LGBMRegressor

from solutions.features import build_features
from solutions.stack import fit_predict_stack, soft_logreg

SEED = 42
MODELS = ["LGBMRegressor", "LogisticRegression"]
SUMMARY = ("kursy + destylacja kursów (średnia LightGBM i LogReg soft z Elo/formy/tabeli/FIFA) "
           "dla lig bez kursów + końcowa LogReg, decyzja argmax p/prior")

__all__ = ["build_features", "fit_predict", "MODELS", "SUMMARY"]

LGB_PARAMS = dict(n_estimators=1000, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.5, reg_lambda=5, random_state=SEED, verbose=-1)


def lgb_distill(Xt, Q, Xv, check=None):
    return np.column_stack([LGBMRegressor(**LGB_PARAMS).fit(Xt, Q[:, j]).predict(Xv) for j in range(3)])


def fit_predict(X_train, y_train, X_test):
    return fit_predict_stack(X_train, y_train, X_test, distillers=[lgb_distill, soft_logreg(0.01)])
