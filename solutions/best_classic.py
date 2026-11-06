"""NAJLEPSZE ROZWIĄZANIE — tor CLASSIC (klasyczny) — wszystko w jednym pliku.

Wynik (walk-forward CV, sezony 2012/13–2014/15): BA = 0.4726 ± 0.0062, holdout 2015/2016: 0.4649.

Podejście:
- cechy (wyłącznie informacje sprzed meczu):
  * kursy 10 bukmacherów → prawdopodobieństwa bez marży (średnia, std), różnica H−A, entropia, flaga braku kursów,
  * ratingi FIFA wyjściowej jedenastki (ostatni pomiar z datą < data meczu): mean/max/min/liczba, potencjał,
    średnie wg formacji (bramkarz/obrona/pomoc/atak z pozycji Y) i różnice gospodarz − gość,
  * kontekst: etap sezonu (stage / liczba kolejek), liga, miesiąc;
- model: imputacja medianą + flagi braków → standaryzacja → LogisticRegression(C=0.003, class_weight="balanced").

Uruchomienie (z katalogu głównego repo):
    uv run python -m solutions.best_classic
Interfejs zgodny z main.py: build_features, fit_predict, MODELS, SUMMARY.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.data import load_table
from common.tracks import assert_classic

SEED = 42
BOOKMAKERS = ("B365", "BW", "IW", "LB", "WH", "VC", "PS", "SJ", "GB", "BS")
MODELS = ["LogisticRegression"]
SUMMARY = ("LogReg(balanced, C=0.003) na: kursy bez marży (śr./std z 10 bukmacherów), "
           "ratingi FIFA wyjściowej 11 (mean/max/min, wg formacji), stage, liga, miesiąc")

__all__ = ["build_features", "fit_predict", "MODELS", "SUMMARY"]


# --------------------------------------------------------------------------- kursy
def odds_features(m: pd.DataFrame) -> pd.DataFrame:
    probs = {r: [] for r in "HDA"}
    for b in BOOKMAKERS:
        cols = [f"{b}{r}" for r in "HDA"]
        if not all(c in m.columns for c in cols):
            continue
        inv = 1.0 / m[cols].where(m[cols] > 1.0)
        s = inv.sum(axis=1, min_count=3)
        for r, c in zip("HDA", cols):
            probs[r].append(inv[c] / s)  # prawdopodobieństwa bez marży
    out = {}
    for r in "HDA":
        P = pd.concat(probs[r], axis=1)
        out[f"odds_p{r}"] = P.mean(axis=1)
        out[f"odds_p{r}_std"] = P.std(axis=1)
    out = pd.DataFrame(out, index=m.index)
    out["odds_pH_minus_pA"] = out["odds_pH"] - out["odds_pA"]
    out["odds_abs_diff"] = out["odds_pH_minus_pA"].abs()
    out["odds_entropy"] = -sum(out[f"odds_p{r}"] * np.log(out[f"odds_p{r}"]) for r in "HDA")
    out["odds_missing"] = out["odds_pH"].isna().astype(float)
    return out


# --------------------------------------------------------------------------- FIFA: gracze
PLAYER_ATTRS = ("overall_rating", "potential")


@lru_cache(maxsize=1)
def _player_attributes() -> pd.DataFrame:
    pa = load_table("Player_Attributes")[["player_api_id", "date", *PLAYER_ATTRS]].dropna()
    pa["date"] = pd.to_datetime(pa["date"])
    return pa.sort_values("date")


def player_features(m: pd.DataFrame) -> pd.DataFrame:
    pa = _player_attributes()
    rows = []
    for side in ("home", "away"):
        for i in range(1, 12):
            rows.append(pd.DataFrame({
                "match_idx": m.index, "side": side, "slot": i,
                "player_api_id": m[f"{side}_player_{i}"], "Y": m[f"{side}_player_Y{i}"], "date": m["date"],
            }))
    lp = pd.concat(rows, ignore_index=True).dropna(subset=["player_api_id"])
    lp["player_api_id"] = lp["player_api_id"].astype(pa["player_api_id"].dtype)
    lp = lp.sort_values("date")
    lp = pd.merge_asof(lp, pa, on="date", by="player_api_id", direction="backward", allow_exact_matches=False)
    # linia: bramkarz (Y=1), obrona (Y<=3), pomoc (Y<=8), atak
    y = lp["Y"]
    lp["line"] = np.select([lp["slot"] == 1, y <= 3, y <= 8], ["gk", "def", "mid"], "att")
    lp.loc[y.isna() & (lp["slot"] != 1), "line"] = "unk"
    out = {}
    g = lp.groupby(["match_idx", "side"])
    agg = g["overall_rating"].agg(["mean", "max", "min", "count"]).unstack("side")
    for stat in ("mean", "max", "min", "count"):
        for side in ("home", "away"):
            out[f"{side}_ovr_{stat}"] = agg[(stat, side)]
    pot = g["potential"].mean().unstack("side")
    out["home_pot_mean"], out["away_pot_mean"] = pot["home"], pot["away"]
    ln = lp.groupby(["match_idx", "side", "line"])["overall_rating"].mean().unstack(["side", "line"])
    for side in ("home", "away"):
        for line in ("gk", "def", "mid", "att"):
            out[f"{side}_ovr_{line}"] = ln.get((side, line))
    out = pd.DataFrame(out).reindex(m.index)
    out["d_ovr_mean"] = out["home_ovr_mean"] - out["away_ovr_mean"]
    out["d_ovr_max"] = out["home_ovr_max"] - out["away_ovr_max"]
    out["d_pot_mean"] = out["home_pot_mean"] - out["away_pot_mean"]
    for line in ("gk", "def", "mid", "att"):
        out[f"d_ovr_{line}"] = out[f"home_ovr_{line}"] - out[f"away_ovr_{line}"]
    out["d_att_vs_def"] = out["home_ovr_att"] - out["away_ovr_def"]
    out["d_def_vs_att"] = out["home_ovr_def"] - out["away_ovr_att"]
    return out


# --------------------------------------------------------------------------- całość
def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"])
    X = pd.concat([odds_features(m), player_features(m)], axis=1)
    X["stage_frac"] = m["stage"] / m.groupby(["league_id", "season"])["stage"].transform("max")
    X["league_id"] = m["league_id"]
    X["month"] = m["date"].dt.month
    return X.astype(float)


# --------------------------------------------------------------------------- model
def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=0.003, class_weight="balanced", max_iter=2000, random_state=SEED),
    )


def fit_predict(X_train, y_train, X_test):
    model = make_model()
    assert_classic(model)
    model.fit(X_train, y_train)
    return model.predict(X_test)


if __name__ == "__main__":
    from common.evaluation import evaluate, make_target

    matches = load_table("Match").sort_values("date").reset_index(drop=True)
    X = build_features(matches)
    print("[classic]", evaluate(fit_predict, X, make_target(matches), matches["season"]))
