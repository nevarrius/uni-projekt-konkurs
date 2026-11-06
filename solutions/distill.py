"""Destylacja kursów bukmacherów — pomysł z agent/nevarious @ 852427d (solutions/stack.py).

Ligi Polski i Szwajcarii (~13% meczów) nie mają żadnych kursów. Na treningowych meczach Z kursami
uczymy model odtwarzać prawdopodobieństwa bukmachera (odds_shinH/D/A) z cech niezależnych od kursów
(Elo, pi-ratings, forma, tabela, FIFA). Cel (rozkład bukmachera) jest dużo mniej zaszumiony niż wynik meczu.
Brakujące kursy (w treningu i teście) zastępujemy destylatem — dla wierszy treningowych bez kursów
jest to predykcja out-of-sample (destylator widzi tylko wiersze z kursami).

Wszystko dzieje się wewnątrz fit_predict (dopasowanie tylko na X_train) — brak wycieku.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from solutions.features import DERIVED_PREFIXES

SHIN = ["odds_shinH", "odds_shinD", "odds_shinA"]
_LABELS = np.array(["H", "D", "A"])

Distiller = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray]


def distill_columns(X: pd.DataFrame) -> list[str]:
    """Cechy bez kursów; bez identyfikatora/statystyk ligi (ligi bez kursów nie występują w treningu destylatora)."""
    return [c for c in X.columns
            if not c.startswith(("odds_", "league_", "abs_odds")) and not c.startswith(DERIVED_PREFIXES)]


def soft_logreg(C: float = 0.01, check: Callable | None = None) -> Distiller:
    """LogisticRegression na miękkich etykietach: każdy wiersz ×3 z wagą = prawdop. bukmachera
    (= entropia krzyżowa względem rozkładu bukmachera)."""

    def fit(Xt: pd.DataFrame, Q: np.ndarray, Xv: pd.DataFrame) -> np.ndarray:
        n = len(Xt)
        Xr = pd.concat([Xt] * 3, ignore_index=True)
        yr = np.repeat(_LABELS, n)
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                              LogisticRegression(C=C, max_iter=3000))
        if check:
            check(model)
        model.fit(Xr, yr, logisticregression__sample_weight=Q.T.ravel())
        P = model.predict_proba(Xv)
        return P[:, [list(model.classes_).index(k) for k in _LABELS]]

    return fit


def fill_odds(X_train: pd.DataFrame, X_test: pd.DataFrame, distillers: list[Distiller]):
    """Zwraca kopie (X_train, X_test) z brakującymi kursami uzupełnionymi destylatem + flagą odds_missing."""
    cols = distill_columns(X_train)
    h_tr = X_train[SHIN].notna().all(axis=1).to_numpy()
    h_te = X_test[SHIN].notna().all(axis=1).to_numpy()
    Xtr, Xte = X_train.copy(), X_test.copy()
    Xtr["odds_missing"] = (~h_tr).astype(float)
    Xte["odds_missing"] = (~h_te).astype(float)
    if h_tr.all() and h_te.all():
        return Xtr, Xte
    target = pd.concat([X_train.loc[~h_tr, cols], X_test.loc[~h_te, cols]])
    Q = X_train.loc[h_tr, SHIN].to_numpy()
    D = np.mean([d(X_train.loc[h_tr, cols], Q, target) for d in distillers], axis=0)
    D = np.clip(D, 0.02, None)
    D /= D.sum(axis=1, keepdims=True)
    n = (~h_tr).sum()
    for frame, mask, part in ((Xtr, ~h_tr, D[:n]), (Xte, ~h_te, D[n:])):
        if not mask.any():
            continue
        idx = frame.index[mask]
        for j, k in enumerate("HDA"):
            for pref in ("odds_shin", "odds_p", "odds_best"):
                frame.loc[idx, f"{pref}{k}"] = part[:, j]
        frame.loc[idx, "odds_logratio"] = np.log(part[:, 0] / part[:, 2])
        frame.loc[idx, "odds_absdiff"] = np.abs(part[:, 0] - part[:, 2])
        frame.loc[idx, "odds_entropy"] = -(part * np.log(part)).sum(axis=1)
        if "abs_odds_logratio" in frame:
            frame.loc[idx, "abs_odds_logratio"] = np.abs(np.log(part[:, 0] / part[:, 2]))
    return Xtr, Xte
