"""Tor CLASSIC — tylko: LogReg, SVM, KNN, drzewa, Bagging, AdaBoost, Random Forest, Voting, sieci neuronowe.

Każdy model MUSI przejść przez assert_classic() przed treningiem.
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.tracks import assert_classic
from solutions import features
from solutions.decision import split_meta

SEED = 42
MODELS = ["LogisticRegression"]
SUMMARY = ("LogReg(balanced, C=0.003) na zwartym zestawie: kursy bez marży (śr./std z 10 bukmacherów), "
           "ratingi FIFA wyjściowej 11 (mean/max/min, wg formacji), stage, liga, miesiąc")

__all__ = ["build_features", "fit_predict", "MODELS", "SUMMARY"]


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    X = features.build_features(matches)
    return X[features.compact_columns(X)]


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=0.003, class_weight="balanced", max_iter=2000, random_state=SEED),
    )


def fit_predict(X_train, y_train, X_test):
    X_train, _ = split_meta(X_train)
    X_test, _ = split_meta(X_test)
    model = make_model()
    assert_classic(model)
    model.fit(X_train, y_train)
    return model.predict(X_test)
