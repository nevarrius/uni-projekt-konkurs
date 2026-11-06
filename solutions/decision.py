"""Reguła decyzyjna pod balanced accuracy: mnożniki prawdopodobieństw klas dobierane
WYŁĄCZNIE na danych treningowych (wewnętrzna walidacja na ostatnim sezonie treningowym).
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

META_COLS = ("season_idx",)
GRID = np.round(np.arange(0.6, 2.01, 0.05), 3)


def split_meta(X: pd.DataFrame):
    return X.drop(columns=[c for c in META_COLS if c in X.columns]), X["season_idx"].to_numpy()


def tune_weights(proba: np.ndarray, y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Mnożniki klas (pierwsza klasa ma wagę 1) maksymalizujące BA na (proba, y)."""
    best, best_w = -1.0, np.ones(len(classes))
    for w in itertools.product(GRID, repeat=len(classes) - 1):
        ww = np.array((1.0, *w))
        s = balanced_accuracy_score(y, classes[np.argmax(proba * ww, axis=1)])
        if s > best:
            best, best_w = s, ww
    return best_w


def fit_predict_tuned(make_model, X_train, y_train, X_test, n_inner: int = 1):
    """make_model() -> nowy (niewytrenowany) estymator z predict_proba.

    1. Dla każdego z n_inner ostatnich sezonów treningowych: trening na wcześniejszych,
       predykcja proba na tym sezonie (walk-forward, bez wycieku).
    2. Mnożniki klas dobrane na zebranych predykcjach.
    3. Model trenowany na całym X_train, predykcja X_test z mnożnikami.
    """
    Xtr, s_tr = split_meta(X_train)
    Xte, _ = split_meta(X_test)
    y_train = np.asarray(y_train)
    seasons = np.unique(s_tr)
    probas, ys = [], []
    for val in seasons[-n_inner:]:
        tr, va = s_tr < val, s_tr == val
        mdl = make_model()
        mdl.fit(Xtr[tr], y_train[tr])
        probas.append(mdl.predict_proba(Xtr[va]))
        ys.append(y_train[va])
        classes = mdl.classes_
    w = tune_weights(np.vstack(probas), np.concatenate(ys), classes)
    mdl = make_model()
    mdl.fit(Xtr, y_train)
    return mdl.classes_[np.argmax(mdl.predict_proba(Xte) * w, axis=1)]
