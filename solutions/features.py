"""Inżynieria cech — wspólna dla obu torów.

Wszystkie cechy używają wyłącznie informacji dostępnych PRZED meczem (AGENTS.md, sekcja 6):
- kursy bukmacherów (publikowane przed meczem),
- Elo / forma / tabela liczone z POPRZEDNICH meczów (bieżący mecz aktualizuje stan dopiero po zapisaniu cech),
- atrybuty FIFA graczy wyjściowej jedenastki i drużyn z ostatniego pomiaru o dacie < data meczu.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from common.data import load_table
from common.evaluation import SEASONS

BOOKMAKERS = ("B365", "BW", "IW", "LB", "WH", "VC", "PS", "SJ", "GB", "BS")


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


# --------------------------------------------------------------------------- Elo
def elo_features(m: pd.DataFrame, k: float = 20.0, home_adv: float = 65.0) -> pd.DataFrame:
    """Elo z mnożnikiem za różnicę bramek; stan zapisywany PRZED aktualizacją bieżącym meczem."""
    order = np.argsort(m["date"].to_numpy(), kind="stable")
    home = m["home_team_api_id"].to_numpy()
    away = m["away_team_api_id"].to_numpy()
    hg = m["home_team_goal"].to_numpy()
    ag = m["away_team_goal"].to_numpy()
    season = m["season"].to_numpy()
    rating: dict[int, float] = {}
    last_season: dict[int, str] = {}
    n_games: dict[int, int] = {}
    eh = np.empty(len(m))
    ea = np.empty(len(m))
    gh = np.empty(len(m))
    ga = np.empty(len(m))
    for i in order:
        h, a = home[i], away[i]
        for t in (h, a):
            if t not in rating:
                rating[t] = 1500.0
                n_games[t] = 0
            elif last_season[t] != season[i]:  # regresja do średniej między sezonami
                rating[t] = 0.8 * rating[t] + 0.2 * 1500.0
            last_season[t] = season[i]
        rh, ra = rating[h], rating[a]
        eh[i], ea[i] = rh, ra
        gh[i], ga[i] = n_games[h], n_games[a]
        exp_h = 1.0 / (1.0 + 10 ** ((ra - rh - home_adv) / 400.0))
        s = 1.0 if hg[i] > ag[i] else 0.5 if hg[i] == ag[i] else 0.0
        gd = abs(hg[i] - ag[i])
        mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
        delta = k * mult * (s - exp_h)
        rating[h] = rh + delta
        rating[a] = ra - delta
        n_games[h] += 1
        n_games[a] += 1
    out = pd.DataFrame(
        {"elo_home": eh, "elo_away": ea, "elo_diff": eh - ea, "games_home": gh, "games_away": ga},
        index=m.index,
    )
    out["elo_exp_home"] = 1.0 / (1.0 + 10 ** (-(out["elo_diff"] + home_adv) / 400.0))
    return out


# --------------------------------------------------------------------------- forma i tabela
def _long_table(m: pd.DataFrame) -> pd.DataFrame:
    base = m[["date", "season", "league_id", "home_team_goal", "away_team_goal"]].copy()
    base["match_idx"] = m.index
    h = base.assign(team=m["home_team_api_id"], is_home=1, gf=m["home_team_goal"], ga=m["away_team_goal"])
    a = base.assign(team=m["away_team_api_id"], is_home=0, gf=m["away_team_goal"], ga=m["home_team_goal"])
    long = pd.concat([h, a], ignore_index=True)
    long["pts"] = np.select([long.gf > long.ga, long.gf == long.ga], [3, 1], 0)
    long["win"] = (long.gf > long.ga).astype(float)
    long["drw"] = (long.gf == long.ga).astype(float)
    long["gd"] = long.gf - long.ga
    return long.sort_values(["date", "match_idx", "is_home"], kind="stable").reset_index(drop=True)


def form_features(m: pd.DataFrame) -> pd.DataFrame:
    long = _long_table(m)
    g = long.groupby("team", sort=False)
    feats = pd.DataFrame(index=long.index)
    for col in ("pts", "gf", "ga", "drw"):
        prev = g[col].shift(1)  # tylko poprzednie mecze
        for n in (5, 10):
            feats[f"{col}_l{n}"] = prev.groupby(long["team"]).transform(lambda s: s.rolling(n, min_periods=1).mean())
        feats[f"{col}_ewm"] = prev.groupby(long["team"]).transform(lambda s: s.ewm(halflife=6, min_periods=1).mean())
    # forma w roli (dom / wyjazd)
    gv = long.groupby(["team", "is_home"], sort=False)
    for col in ("pts", "gd"):
        prev = gv[col].shift(1)
        feats[f"{col}_venue_l6"] = prev.groupby([long["team"], long["is_home"]]).transform(
            lambda s: s.rolling(6, min_periods=1).mean()
        )
    # dni odpoczynku
    feats["rest_days"] = g["date"].diff().dt.days.clip(upper=30)
    # tabela w bieżącym sezonie przed meczem
    gs = long.groupby(["team", "season"], sort=False)
    n_prev = gs.cumcount()
    feats["season_ppg"] = (gs["pts"].cumsum() - long["pts"]) / n_prev.replace(0, np.nan)
    feats["season_gdpg"] = (gs["gd"].cumsum() - long["gd"]) / n_prev.replace(0, np.nan)
    feats["season_played"] = n_prev
    # pozycja w tabeli ligi przed meczem: ranking punktów sezonowych w danym dniu
    long["cum_pts"] = gs["pts"].cumsum() - long["pts"]
    feats["table_rank_pct"] = long.groupby(["league_id", "season", "date"])["cum_pts"].rank(pct=True)

    feats["match_idx"] = long["match_idx"]
    feats["is_home"] = long["is_home"]
    fh = feats[feats.is_home == 1].set_index("match_idx").drop(columns="is_home").add_prefix("h_")
    fa = feats[feats.is_home == 0].set_index("match_idx").drop(columns="is_home").add_prefix("a_")
    out = fh.join(fa).reindex(m.index)
    for col in ("pts_l5", "pts_l10", "pts_ewm", "gf_ewm", "ga_ewm", "season_ppg", "season_gdpg", "rest_days"):
        out[f"d_{col}"] = out[f"h_{col}"] - out[f"a_{col}"]
    out["d_venue_pts"] = out["h_pts_venue_l6"] - out["a_pts_venue_l6"]

    # bilans bezpośrednich spotkań (z perspektywy gospodarza, poprzednie mecze)
    key = np.where(
        m["home_team_api_id"] < m["away_team_api_id"],
        m["home_team_api_id"].astype(str) + "_" + m["away_team_api_id"].astype(str),
        m["away_team_api_id"].astype(str) + "_" + m["home_team_api_id"].astype(str),
    )
    sign = np.where(m["home_team_api_id"] < m["away_team_api_id"], 1.0, -1.0)
    h2h = pd.DataFrame({"key": key, "date": m["date"], "gd": (m["home_team_goal"] - m["away_team_goal"]) * sign,
                        "drw": (m["home_team_goal"] == m["away_team_goal"]).astype(float)}, index=m.index)
    h2h = h2h.sort_values("date", kind="stable")
    gk = h2h.groupby("key")
    h2h["h2h_gd"] = gk["gd"].transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    h2h["h2h_drw"] = gk["drw"].transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    h2h["h2h_n"] = gk.cumcount()
    h2h = h2h.reindex(m.index)
    out["h2h_gd"] = h2h["h2h_gd"] * sign
    out["h2h_drw"] = h2h["h2h_drw"]
    out["h2h_n"] = h2h["h2h_n"]
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


# --------------------------------------------------------------------------- FIFA: drużyny
TEAM_ATTRS = ("buildUpPlaySpeed", "buildUpPlayPassing", "chanceCreationPassing", "chanceCreationCrossing",
              "chanceCreationShooting", "defencePressure", "defenceAggression", "defenceTeamWidth")


def team_features(m: pd.DataFrame) -> pd.DataFrame:
    ta = load_table("Team_Attributes")[["team_api_id", "date", *TEAM_ATTRS]].copy()
    ta["date"] = pd.to_datetime(ta["date"])
    ta = ta.sort_values("date")
    out = pd.DataFrame(index=m.index)
    for side in ("home", "away"):
        left = pd.DataFrame({"date": m["date"], "team_api_id": m[f"{side}_team_api_id"], "idx": m.index})
        left = left.sort_values("date")
        mg = pd.merge_asof(left, ta, on="date", by="team_api_id", direction="backward", allow_exact_matches=False)
        mg = mg.set_index("idx").reindex(m.index)
        for c in TEAM_ATTRS:
            out[f"{side}_{c}"] = mg[c]
    for c in TEAM_ATTRS:
        out[f"d_{c}"] = out[f"home_{c}"] - out[f"away_{c}"]
    return out


# --------------------------------------------------------------------------- dynamiczny model Poissona
def poisson_features(m: pd.DataFrame, alpha: float = 0.07, alpha_lg: float = 0.01) -> pd.DataFrame:
    """Wygładzane wykładniczo siły ataku/obrony (osobno dom/wyjazd) → λ goli → P(H/D/A) z rozkładu Poissona.

    Stan zapisywany PRZED aktualizacją bieżącym meczem (bez wycieku)."""
    from math import exp, factorial

    order = np.argsort(m["date"].to_numpy(), kind="stable")
    home, away = m["home_team_api_id"].to_numpy(), m["away_team_api_id"].to_numpy()
    hg, ag = m["home_team_goal"].to_numpy(float), m["away_team_goal"].to_numpy(float)
    lg = m["league_id"].to_numpy()
    lg_h: dict = {}
    lg_a: dict = {}
    lg_d: dict = {}
    att_h: dict = {}
    def_h: dict = {}
    att_a: dict = {}
    def_a: dict = {}
    n = len(m)
    lam_h, lam_a, lgd = np.empty(n), np.empty(n), np.empty(n)
    for i in order:
        L, h, a = lg[i], home[i], away[i]
        if L not in lg_h:
            lg_h[L], lg_a[L], lg_d[L] = 1.5, 1.1, 0.26
        lh, la = lg_h[L], lg_a[L]
        att_h.setdefault(h, lh); def_h.setdefault(h, la)
        att_a.setdefault(a, la); def_a.setdefault(a, lh)
        lam_h[i] = att_h[h] * def_a[a] / lh
        lam_a[i] = att_a[a] * def_h[h] / la
        lgd[i] = lg_d[L]
        # aktualizacja po meczu
        att_h[h] += alpha * (hg[i] - att_h[h]); def_h[h] += alpha * (ag[i] - def_h[h])
        att_a[a] += alpha * (ag[i] - att_a[a]); def_a[a] += alpha * (hg[i] - def_a[a])
        lg_h[L] += alpha_lg * (hg[i] - lh); lg_a[L] += alpha_lg * (ag[i] - la)
        lg_d[L] += alpha_lg * (float(hg[i] == ag[i]) - lg_d[L])
    k = np.arange(11)
    fact = np.array([factorial(int(x)) for x in k], dtype=float)
    ph = np.exp(-lam_h)[:, None] * lam_h[:, None] ** k / fact
    pa = np.exp(-lam_a)[:, None] * lam_a[:, None] ** k / fact
    joint = ph[:, :, None] * pa[:, None, :]
    p_home = np.tril(np.ones((11, 11)), -1)
    pH = (joint * p_home).sum((1, 2))
    pD = np.einsum("nii->n", joint)
    pA = (joint * p_home.T).sum((1, 2))
    tot = pH + pD + pA
    return pd.DataFrame({
        "pois_lam_h": lam_h, "pois_lam_a": lam_a, "pois_lam_diff": lam_h - lam_a,
        "pois_pH": pH / tot, "pois_pD": pD / tot, "pois_pA": pA / tot, "league_draw_rate": lgd,
    }, index=m.index)


# --------------------------------------------------------------------------- całość
def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"])
    parts = [
        odds_features(m),
        elo_features(m),
        form_features(m),
        player_features(m),
        team_features(m),
        poisson_features(m),
    ]
    X = pd.concat(parts, axis=1)
    # bliskość sił (remis) — nieliniowość dla modeli liniowych
    X["abs_elo"] = (X["elo_diff"] + 65.0).abs()
    X["abs_ovr"] = X["d_ovr_mean"].abs()
    X["abs_ppg"] = X["d_season_ppg"].abs()
    X["abs_pts_ewm"] = X["d_pts_ewm"].abs()
    X["abs_pois"] = X["pois_lam_diff"].abs()
    X["stage_frac"] = m["stage"] / m.groupby(["league_id", "season"])["stage"].transform("max")
    X["league_id"] = m["league_id"]
    X["month"] = m["date"].dt.month
    X["season_idx"] = m["season"].map({s: i for i, s in enumerate(SEASONS)})  # meta, usuwana przed modelem
    return X.astype(float)


def compact_columns(X: pd.DataFrame) -> list[str]:
    """Zwarty zestaw: kursy + ratingi FIFA wyjściowej 11 + kontekst (najlepszy w przeszukaniu podzbiorów grup na CV).

    Elo / forma / tabela / atrybuty drużyn są w dużej mierze redundantne względem kursów i dokładają szum."""
    keep = []
    for c in X.columns:
        if c.startswith("odds_") or c in ("stage_frac", "league_id", "month", "season_idx"):
            keep.append(c)
        elif "_ovr_" in c or c.startswith("d_ovr") or "pot_mean" in c or c in ("d_att_vs_def", "d_def_vs_att"):
            keep.append(c)
    return keep
