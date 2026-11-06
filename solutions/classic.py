"""Tor CLASSIC — tylko: LogReg, SVM, KNN, drzewa, Bagging, AdaBoost, Random Forest, Voting, sieci neuronowe.

Każdy model MUSI przejść przez assert_classic() przed treningiem.
"""

from itertools import product

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.tracks import assert_classic
from solutions.features import build_raw_odds_features as build_features

SEED = 42
MODELS = ["LogisticRegression"]  # zadeklaruj użyte modele (trafia do SCORE.json)
SUMMARY = "LogReg(balanced) z progami dostrojonymi na końcowym fragmencie treningu"

__all__ = ["build_features", "fit_predict", "MODELS", "SUMMARY"]


def fit_predict(X_train, y_train, X_test):
    def make_model():
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
        )

    validation_size = max(500, int(len(X_train) * 0.2))
    split = len(X_train) - validation_size
    tuning_model = make_model()
    assert_classic(tuning_model)
    tuning_model.fit(X_train.iloc[:split], y_train.iloc[:split])
    validation_probabilities = tuning_model.predict_proba(X_train.iloc[split:])

    best_score = -1.0
    best_weights = np.ones(len(tuning_model.classes_))
    for weights in product((0.6, 0.8, 1.0, 1.25, 1.5), repeat=len(tuning_model.classes_)):
        weights_array = np.asarray(weights)
        predictions = tuning_model.classes_[np.argmax(validation_probabilities * weights_array, axis=1)]
        score = balanced_accuracy_score(y_train.iloc[split:], predictions)
        if score > best_score:
            best_score = score
            best_weights = weights_array

    model = make_model()
    assert_classic(model)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    return model.classes_[np.argmax(probabilities * best_weights, axis=1)]
