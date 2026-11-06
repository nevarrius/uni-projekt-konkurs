"""Tor CLASSIC — tylko: LogReg, SVM, KNN, drzewa, Bagging, AdaBoost, Random Forest, Voting, sieci neuronowe.

Każdy model MUSI przejść przez assert_classic() przed treningiem.
"""

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.tracks import assert_classic
from solutions.features import build_features

SEED = 42
MODELS = ["LogisticRegression"]  # zadeklaruj użyte modele (trafia do SCORE.json)
SUMMARY = "LogReg(balanced) na implikowanych prawdop. z kursów bukmacherów"

__all__ = ["build_features", "fit_predict", "MODELS", "SUMMARY"]


def fit_predict(X_train, y_train, X_test):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
    )
    assert_classic(model)
    model.fit(X_train, y_train)
    return model.predict(X_test)
