"""Tor CLASSIC — tylko: LogReg, SVM, KNN, drzewa, Bagging, AdaBoost, Random Forest, Voting, sieci neuronowe.

Podejście:
  * cechy z solutions/features.py (kursy z usuniętą marżą, Elo, pi-ratings, forma, tabela, H2H,
    atrybuty FIFA składów i drużyn) — bez cech pochodnych (abs_/pois_), które w CV nie pomagały,
  * soft VotingClassifier (równe wagi): LogisticRegression(C=0.05) + RandomForest(500, leaf=30)
    + MLP(16, alpha=10) — każdy we własnym Pipeline (imputacja medianą, standaryzacja),
  * reguła decyzyjna pod balanced accuracy: argmax_k P(k|x) / prior_k (prior z danych treningowych) —
    dla skalibrowanych prawdopodobieństw to decyzja maksymalizująca oczekiwaną BA.

Hybryda z destylacją kursów (pomysł: agent/nevarious @ 852427d, solutions/distill.py):
  * brakujące kursy (ligi PL/CH) zastępujemy destylatem — LogReg na miękkich etykietach bukmachera,
  * mecze Z kursami:  P = 0.5 · LogReg(log kursów Shina)  +  0.5 · voting,
  * mecze BEZ kursów: P = voting (na cechach z destylowanymi kursami).

Każdy model MUSI przejść przez assert_classic() przed treningiem.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.evaluation import CLASSES
from common.tracks import assert_classic
from solutions.distill import SHIN, fill_odds, soft_logreg
from solutions.features import DERIVED_PREFIXES, build_features

SEED = 42
MODELS = ["VotingClassifier", "LogisticRegression", "RandomForestClassifier", "MLPClassifier"]
SUMMARY = ("hybryda: z kursami 0.5·LogReg(log-kursy)+0.5·voting, bez kursów voting na destylowanych kursach "
           "(destylacja wg agent/nevarious); soft voting LogReg+RF+MLP na kursach (Shin), Elo, pi-ratings, formie, tabeli, H2H, atrybutach FIFA składów/drużyn; "
           "decyzja argmax p/prior pod BA")

__all__ = ["build_features", "fit_predict", "MODELS", "SUMMARY"]


def _cols(X):
    return [c for c in X.columns if not c.startswith(DERIVED_PREFIXES)]


def predict_proba(X_train, y_train, X_test) -> np.ndarray:
    cols = _cols(X_train)
    model = VotingClassifier(
        [
            ("lr", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 LogisticRegression(C=0.05, max_iter=3000, random_state=SEED))),
            ("rf", make_pipeline(SimpleImputer(strategy="median"),
                                 RandomForestClassifier(500, min_samples_leaf=30, max_features=0.3,
                                                        n_jobs=-1, random_state=SEED))),
            ("mlp", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  MLPClassifier((16,), alpha=10.0, max_iter=500, early_stopping=True,
                                                random_state=SEED))),
        ],
        voting="soft",
    )
    assert_classic(model)
    model.fit(X_train[cols], y_train)
    P = model.predict_proba(X_test[cols])
    return P[:, [list(model.classes_).index(c) for c in CLASSES]]


def decide(P: np.ndarray, y_train) -> np.ndarray:
    """argmax p_k / prior_k — decyzja optymalna dla balanced accuracy."""
    prior = np.array([(np.asarray(y_train) == c).mean() for c in CLASSES])
    return np.array(CLASSES)[(P / prior).argmax(axis=1)]


def market_proba(X_train, y_train, X_test) -> np.ndarray:
    """LogReg na log(kursów Shina) — kalibracja rynku, uczona na treningowych meczach z kursami."""
    has = X_train[SHIN].notna().all(axis=1).to_numpy()
    model = LogisticRegression(C=1.0, max_iter=2000)
    assert_classic(model)
    model.fit(np.log(X_train.loc[has, SHIN]), np.asarray(y_train)[has])
    P = model.predict_proba(np.log(X_test[SHIN].fillna(1 / 3)))
    return P[:, [list(model.classes_).index(c) for c in CLASSES]]


def hybrid_proba(X_train, y_train, X_test, market, alpha: float) -> np.ndarray:
    """Na meczach z kursami: alpha·market + (1−alpha)·voting; bez kursów: voting na destylowanych kursach."""
    has_te = X_test[SHIN].notna().all(axis=1).to_numpy()[:, None]
    Xtr, Xte = fill_odds(X_train, X_test, [soft_logreg(0.01, check=assert_classic)])
    P_vote = predict_proba(Xtr, y_train, Xte)
    return np.where(has_te, alpha * market + (1 - alpha) * P_vote, P_vote)


def fit_predict(X_train, y_train, X_test):
    P = hybrid_proba(X_train, y_train, X_test, market_proba(X_train, y_train, X_test), alpha=0.5)
    return decide(P, y_train)
