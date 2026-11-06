"""Tor CLASSIC — tylko: LogReg, SVM, KNN, drzewa, Bagging, AdaBoost, Random Forest, Voting, sieci neuronowe.

Każdy model przechodzi przez assert_classic() przed treningiem (przekazywany jako `check`).
Architektura: solutions/stack.py (kursy + destylacja kursów regresją logistyczną + końcowa LogReg).
"""

from common.tracks import assert_classic
from solutions.features import build_features
from solutions.stack import fit_predict_stack, soft_logreg

MODELS = ["LogisticRegression"]
SUMMARY = ("kursy + destylacja kursów (LogReg na miękkich etykietach z Elo/formy/tabeli/FIFA) "
           "dla lig bez kursów + końcowa LogReg, decyzja argmax p/prior")

__all__ = ["build_features", "fit_predict", "MODELS", "SUMMARY"]


def fit_predict(X_train, y_train, X_test):
    return fit_predict_stack(X_train, y_train, X_test, distillers=[soft_logreg(0.01)], check=assert_classic)
