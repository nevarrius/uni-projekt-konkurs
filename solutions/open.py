"""Tor OPEN — dowolne metody (gradient boosting, stacking, własne sieci, ensembling czegokolwiek)."""

from sklearn.ensemble import HistGradientBoostingClassifier

from solutions.features import build_features

SEED = 42
MODELS = ["HistGradientBoostingClassifier"]  # zadeklaruj użyte modele (trafia do SCORE.json)
SUMMARY = "HistGradientBoosting(balanced) na implikowanych prawdop. z kursów bukmacherów"

__all__ = ["build_features", "fit_predict", "MODELS", "SUMMARY"]


def fit_predict(X_train, y_train, X_test):
    model = HistGradientBoostingClassifier(
        class_weight="balanced", max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=SEED
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)
