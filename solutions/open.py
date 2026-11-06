"""Tor OPEN — dowolne metody (gradient boosting, stacking, własne sieci, ensembling czegokolwiek)."""

from sklearn.ensemble import HistGradientBoostingClassifier

from solutions.decision import split_meta
from solutions.features import build_features

SEED = 42
MODELS = ["HistGradientBoostingClassifier"]
SUMMARY = ("HistGradientBoosting(balanced) na: kursach bez marży, Elo, formie, tabeli, H2H, "
           "ratingach FIFA składów (wg formacji), atrybutach taktycznych drużyn")

__all__ = ["build_features", "fit_predict", "MODELS", "SUMMARY"]


def fit_predict(X_train, y_train, X_test):
    X_train, _ = split_meta(X_train)
    X_test, _ = split_meta(X_test)
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=100, max_features=0.5,
        l2_regularization=5, class_weight="balanced", random_state=SEED,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)
