"""Tor OPEN — hybryda: rynek (surowe kursy Shina) + soft voting z toru classic, z destylacją kursów.

  * mecze Z kursami:  P = 0.75 · prawdopodobieństwa bukmachera (Shin, bez modelu)  +  0.25 · voting,
  * mecze BEZ kursów (ligi PL/CH): P = voting LogReg+RF+MLP na cechach z kursami destylowanymi
    (LogReg na miękkich etykietach bukmachera — pomysł agent/nevarious @ 852427d),
  * decyzja argmax p_k / prior_k.
Wcześniejszy blend z CatBoost/HistGB okazał się gorszy po dodaniu rynku (EXPERIMENTS.md).
W torze classic rynek przechodzi przez kalibrującą LogReg; tu (dowolne metody) używamy go wprost.
"""

from solutions import classic
from solutions.distill import SHIN
from solutions.features import build_features

SEED = 42
MODELS = ["VotingClassifier", "LogisticRegression", "RandomForestClassifier", "MLPClassifier"]
SUMMARY = ("hybryda: z kursami 0.75·kursy(Shin)+0.25·voting, bez kursów voting LogReg+RF+MLP na destylowanych "
           "kursach (destylacja wg agent/nevarious); Elo, pi-ratings, forma, tabela, H2H, FIFA; decyzja p/prior")

__all__ = ["build_features", "fit_predict", "MODELS", "SUMMARY"]


def fit_predict(X_train, y_train, X_test):
    market = X_test[SHIN].to_numpy()  # NaN tam, gdzie brak kursów — te wiersze i tak biorą voting
    P = classic.hybrid_proba(X_train, y_train, X_test, market, alpha=0.75)
    return classic.decide(P, y_train)
