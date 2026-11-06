"""Wspólny protokół ewaluacji konkursu — NIE MODYFIKOWAĆ na własnych branchach.

Zadanie: klasyfikacja wyniku meczu z perspektywy GOSPODARZA (home team) do 3 klas:
    "win"  — gospodarz wygrał  (home_team_goal > away_team_goal)
    "draw" — remis
    "loss" — gospodarz przegrał
Metryka: balanced accuracy (średni recall klas).

Podział czasowy (po sezonach, starsze → trening, nowsze → walidacja/test):
    TEST (holdout):  2015/2016 — liczony raz, model trenowany na wszystkich sezonach ≤ 2014/2015
    CV (walk-forward, expanding window) — 3 foldy:
        fold 1: trening 2008/09–2011/12  → walidacja 2012/2013
        fold 2: trening 2008/09–2012/13  → walidacja 2013/2014
        fold 3: trening 2008/09–2013/14  → walidacja 2014/2015

Użycie:
    from common.evaluation import make_target, evaluate

    y = make_target(matches)                    # Series z etykietami, indeks jak matches
    result = evaluate(fit_predict, X, y, matches["season"])
    print(result)                               # cv_score, cv_std, holdout_score, foldy

gdzie `fit_predict(X_train, y_train, X_test) -> y_pred` trenuje model OD ZERA
na X_train i zwraca etykiety dla X_test. Wszystko, co jest uczone na danych
(imputacja, skalowanie, encodery, selekcja cech, tuning), musi dziać się wewnątrz
fit_predict — inaczej wynik jest zawyżony przez wyciek.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

CLASSES = ("win", "draw", "loss")
METRIC = "balanced_accuracy"

SEASONS = (
    "2008/2009",
    "2009/2010",
    "2010/2011",
    "2011/2012",
    "2012/2013",
    "2013/2014",
    "2014/2015",
    "2015/2016",
)
TEST_SEASON = "2015/2016"
CV_VAL_SEASONS = ("2012/2013", "2013/2014", "2014/2015")

FitPredict = Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray]


def make_target(matches: pd.DataFrame) -> pd.Series:
    """Etykieta wyniku z perspektywy gospodarza na podstawie surowej tabeli Match."""
    h, a = matches["home_team_goal"], matches["away_team_goal"]
    y = np.select([h > a, h < a], ["win", "loss"], default="draw")
    return pd.Series(y, index=matches.index, name="result")


def cv_splits(season: pd.Series):
    """Generator (nazwa_sezonu_walidacyjnego, maska_train, maska_val) — walk-forward po sezonach."""
    order = {s: i for i, s in enumerate(SEASONS)}
    idx = season.map(order)
    if idx.isna().any():
        raise ValueError(f"Nieznane sezony: {sorted(season[idx.isna()].unique())}")
    for val in CV_VAL_SEASONS:
        yield val, (idx < order[val]).to_numpy(), (idx == order[val]).to_numpy()


def holdout_split(season: pd.Series):
    """(maska_train, maska_test): trening = wszystkie sezony przed TEST_SEASON."""
    order = {s: i for i, s in enumerate(SEASONS)}
    idx = season.map(order)
    return (idx < order[TEST_SEASON]).to_numpy(), (idx == order[TEST_SEASON]).to_numpy()


@dataclass
class Result:
    cv_score: float
    cv_std: float
    holdout_score: float | None
    folds: dict[str, float]

    def __str__(self) -> str:
        folds = ", ".join(f"{k}: {v:.4f}" for k, v in self.folds.items())
        ho = "-" if self.holdout_score is None else f"{self.holdout_score:.4f}"
        return f"CV BA = {self.cv_score:.4f} ± {self.cv_std:.4f}  [{folds}]  |  holdout {TEST_SEASON} BA = {ho}"

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(
    fit_predict: FitPredict,
    X: pd.DataFrame,
    y: pd.Series,
    season: pd.Series,
    holdout: bool = True,
) -> Result:
    """Liczy BA w walk-forward CV oraz (opcjonalnie) na holdoucie 2015/2016.

    X, y, season muszą mieć ten sam indeks/kolejność wierszy (jeden wiersz = jeden mecz).
    Wiersze można odfiltrować (np. mecze bez danych), ale wtedy wynik nie jest porównywalny —
    regulamin wymaga predykcji dla WSZYSTKICH meczów sezonu walidacyjnego/testowego.
    """
    if not (len(X) == len(y) == len(season)):
        raise ValueError("X, y i season muszą mieć tę samą długość")
    y = pd.Series(np.asarray(y), index=X.index)
    season = pd.Series(np.asarray(season), index=X.index)

    folds = {}
    for name, tr, va in cv_splits(season):
        pred = fit_predict(X[tr], y[tr], X[va])
        folds[name] = float(balanced_accuracy_score(y[va], pred))
    scores = np.array(list(folds.values()))

    holdout_score = None
    if holdout:
        tr, te = holdout_split(season)
        pred = fit_predict(X[tr], y[tr], X[te])
        holdout_score = float(balanced_accuracy_score(y[te], pred))

    return Result(float(scores.mean()), float(scores.std()), holdout_score, folds)
