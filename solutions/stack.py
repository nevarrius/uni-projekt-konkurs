"""Architektura "kursy + destylacja" wspólna dla obu torów.

Idea:
  1. Kursy bukmacherów to najlepsza dostępna informacja (BA ≈ 0.48 przy regule argmax p/prior),
     ale w ligach Polski i Szwajcarii (~13% meczów) kursów nie ma wcale.
  2. DESTYLACJA: na treningowych meczach z kursami uczymy model odtwarzać prawdopodobieństwa
     bukmacherów (pD, pA, pH) z cech niezależnych od kursów (Elo, forma, tabela, FIFA składów).
     Cel jest dużo mniej zaszumiony niż sam wynik meczu → lepszy model "bez kursów".
  3. q = kursy, jeśli są; w przeciwnym razie destylowane prawdopodobieństwa.
  4. Końcowa regresja logistyczna na log(q) z interakcjami "brak kursów" — kalibruje osobno
     destylację dla lig bez kursów. Wiersze treningowe bez kursów dostają predykcję destylacji
     out-of-sample (destylacja jest trenowana tylko na wierszach z kursami).
  5. Decyzja: argmax P(c) / prior(c) (prior z treningu) — reguła optymalna dla balanced accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = np.array(["draw", "loss", "win"])  # kolejność sklearn (alfabetyczna)
ODDS_P = ["odds_pD", "odds_pA", "odds_pH"]  # ta sama kolejność co CLASSES
SEED = 42


def non_odds_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if not c.startswith(("odds_", "league_"))]


def _normalize(R: np.ndarray) -> np.ndarray:
    R = np.clip(R, 0.02, None)
    return R / R.sum(axis=1, keepdims=True)


def soft_logreg(C: float = 0.01):
    """Destylator z dozwolonych modeli: LogisticRegression na miękkich etykietach.

    Każdy wiersz powielony 3× (po jednym na klasę) z wagą = prawdopodobieństwo bukmachera,
    co daje dokładnie entropię krzyżową względem rozkładu bukmachera.
    """

    def distill(Xt: pd.DataFrame, Q: np.ndarray, Xv: pd.DataFrame, check=None) -> np.ndarray:
        n = len(Xt)
        Xr = pd.concat([Xt] * 3, ignore_index=True)
        yr = np.repeat(CLASSES, n)
        wr = Q.T.ravel()
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=C, max_iter=3000),
        )
        if check:
            check(model)
        model.fit(Xr, yr, logisticregression__sample_weight=wr)
        assert list(model.classes_) == list(CLASSES)
        return model.predict_proba(Xv)

    return distill


def _final_design(q: np.ndarray, has_odds: np.ndarray) -> np.ndarray:
    L = np.log(q)
    no = (~has_odds).astype(float)[:, None]
    return np.hstack([L, L * no, no])


def fit_predict_stack(X_train, y_train, X_test, distillers, check=None, hybrid=False):
    """distillers: lista funkcji (Xt, Q, Xv, check) -> prawdopodobieństwa; wynik uśredniany."""
    y_train = np.asarray(y_train)
    cols = non_odds_columns(X_train)
    Q_tr = X_train[ODDS_P].to_numpy()
    Q_te = X_test[ODDS_P].to_numpy()
    h_tr = ~np.isnan(Q_tr).any(axis=1)
    h_te = ~np.isnan(Q_te).any(axis=1)

    # destylacja: trenowana na wierszach treningowych z kursami,
    # przewiduje dla treningowych bez kursów (out-of-sample) i dla całego testu
    target = pd.concat([X_train.loc[~h_tr, cols], X_test[cols]])
    D = np.mean([_normalize(d(X_train.loc[h_tr, cols], Q_tr[h_tr], target, check)) for d in distillers], axis=0)
    n_tr_no = (~h_tr).sum()
    q_tr = Q_tr.copy(); q_tr[~h_tr] = D[:n_tr_no]
    q_te = Q_te.copy(); q_te[~h_te] = D[n_tr_no:][~h_te]
    q_tr, q_te = _normalize(q_tr), _normalize(q_te)

    final = LogisticRegression(C=1.0, max_iter=2000)
    if check:
        check(final)
    final.fit(_final_design(q_tr, h_tr), y_train)
    assert list(final.classes_) == list(CLASSES)
    P = final.predict_proba(_final_design(q_te, h_te))
    if hybrid:
        P[h_te] = q_te[h_te]

    prior = pd.Series(y_train).value_counts(normalize=True).reindex(CLASSES).to_numpy()
    return CLASSES[(P / prior).argmax(axis=1)]
