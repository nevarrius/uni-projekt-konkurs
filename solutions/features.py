"""Inżynieria cech — wspólna dla obu torów.

Wszystkie cechy używają WYŁĄCZNIE informacji dostępnych przed meczem (AGENTS.md, sekcja 6):
  * kursy bukmacherów (publikowane przed meczem) — z usuniętą marżą (normalizacja + metoda Shina),
  * ratingi drużyn liczone sekwencyjnie po dacie: Elo (z marginesem bramek) i pi-ratings
    (Constantinou & Fenton 2013) — zapisujemy stan PRZED aktualizacją bieżącym meczem,
  * forma z poprzednich meczów (shift), tabela ligowa przed kolejką, dni odpoczynku, bilans H2H,
  * atrybuty FIFA wyjściowych składów — ostatni pomiar z datą < data meczu (merge_asof, strict),
  * atrybuty taktyczne drużyn (Team_Attributes) — ostatni pomiar z datą < data meczu,
  * statystyki z XML (strzały celne, posiadanie) — wyłącznie jako średnie z poprzednich meczów.
Mecze z tego samego dnia nie widzą nawzajem swoich wyników (grupowanie po dacie przy ratingach).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common.data import load_table

DERIVED_PREFIXES = ("abs_", "pois_", "exp_total_goals", "draw_tendency")

BOOKMAKERS = ("B365", "BW", "IW", "LB", "PS", "WH", "SJ", "VC", "GB", "BS")

_CACHE: dict[int, pd.DataFrame] = {}


# --------------------------------------------------------------------------------------
# Kursy bukmacherów
# --------------------------------------------------------------------------------------
def _shin(pi: np.ndarray, iters: int = 30) -> np.ndarray:
    """Metoda Shina (1993) usuwania marży — iteracja punktu stałego po z (udział insiderów).

    pi: macierz n×3 surowych odwrotności kursów (suma > 1).
    """
    s = pi.sum(axis=1, keepdims=True)
    z = np.zeros((len(pi), 1))
    for _ in range(iters):
        p = (np.sqrt(z**2 + 4 * (1 - z) * pi**2 / s) - z) / (2 * (1 - z))
        z = (p.sum(axis=1, keepdims=True) - 1) / (pi.shape[1] - 2)
        z = np.clip(z, 0, 0.5)
    p = (np.sqrt(z**2 + 4 * (1 - z) * pi**2 / s) - z) / (2 * (1 - z))
    return p / p.sum(axis=1, keepdims=True)


def odds_features(m: pd.DataFrame) -> pd.DataFrame:
    probs = []  # lista macierzy n×3 (znormalizowane prawdopodobieństwa) per bukmacher
    raws = []
    for b in BOOKMAKERS:
        o = m[[f"{b}H", f"{b}D", f"{b}A"]].to_numpy(float, copy=True)
        o[(o <= 1.0) | ~np.isfinite(o)] = np.nan
        r = 1.0 / o
        r[np.isnan(r).any(axis=1)] = np.nan
        raws.append(r)
        probs.append(r / r.sum(axis=1, keepdims=True))
    P = np.stack(probs)  # B×n×3
    R = np.stack(raws)
    out = pd.DataFrame(index=m.index)
    with np.errstate(all="ignore"), pd.option_context("mode.chained_assignment", None):
        import warnings

        warnings.simplefilter("ignore", RuntimeWarning)
        mean_p = np.nanmean(P, axis=0)
        std_p = np.nanstd(P, axis=0)
        mean_raw = np.nanmean(R, axis=0)
        # najlepsze kursy na rynku (min odwrotność) → prawdop. o najmniejszej marży
        best = np.nanmin(R, axis=0)
        n_books = np.sum(~np.isnan(R[:, :, 0]), axis=0)
    shin = np.full_like(mean_raw, np.nan)
    ok = ~np.isnan(mean_raw).any(axis=1)
    shin[ok] = _shin(mean_raw[ok])
    for j, k in enumerate("HDA"):
        out[f"odds_p{k}"] = mean_p[:, j]
        out[f"odds_shin{k}"] = shin[:, j]
        out[f"odds_std{k}"] = std_p[:, j]
        out[f"odds_best{k}"] = best[:, j] / np.nansum(best, axis=1)
    out["odds_margin"] = mean_raw.sum(axis=1) - 1
    out["odds_n_books"] = n_books
    out["odds_logratio"] = np.log(out["odds_shinH"] / out["odds_shinA"])
    out["odds_absdiff"] = (out["odds_shinH"] - out["odds_shinA"]).abs()
    out["odds_entropy"] = -(shin * np.log(shin)).sum(axis=1)
    out.loc[~ok, "odds_entropy"] = np.nan
    return out


# --------------------------------------------------------------------------------------
# Ratingi: Elo i pi-ratings (sekwencyjnie po dniach)
# --------------------------------------------------------------------------------------
def rating_features(m: pd.DataFrame) -> pd.DataFrame:
    n = len(m)
    home = m["home_team_api_id"].to_numpy()
    away = m["away_team_api_id"].to_numpy()
    hg = m["home_team_goal"].to_numpy(float, copy=True)
    ag = m["away_team_goal"].to_numpy(float, copy=True)
    dates = m["date"].to_numpy()
    season = m["season"].to_numpy()

    elo: dict[int, float] = {}
    pi_h: dict[int, float] = {}  # pi-rating "u siebie"
    pi_a: dict[int, float] = {}  # pi-rating "na wyjeździe"
    last_season: dict[int, str] = {}
    K, HFA = 20.0, 60.0
    LAM, GAM, C = 0.06, 0.5, 3.0

    out = np.zeros((n, 6))
    order = np.argsort(dates, kind="stable")
    # grupowanie po dacie: najpierw zapisz stan przed wszystkimi meczami dnia, potem aktualizuj
    i = 0
    while i < n:
        j = i
        while j < n and dates[order[j]] == dates[order[i]]:
            j += 1
        idx = order[i:j]
        for t in idx:
            for team in (home[t], away[t]):
                if team in last_season and last_season[team] != season[t]:
                    elo[team] = 1500 + 0.8 * (elo[team] - 1500)  # regresja do średniej między sezonami
                last_season[team] = season[t]
            eh, ea = elo.get(home[t], 1500.0), elo.get(away[t], 1500.0)
            out[t, 0], out[t, 1] = eh, ea
            out[t, 2], out[t, 3] = pi_h.get(home[t], 0.0), pi_a.get(away[t], 0.0)
            out[t, 4], out[t, 5] = pi_a.get(home[t], 0.0), pi_h.get(away[t], 0.0)
        for t in idx:
            h, a = home[t], away[t]
            # Elo z mnożnikiem marginesu (FiveThirtyEight/World Football Elo)
            eh, ea = elo.get(h, 1500.0), elo.get(a, 1500.0)
            exp_h = 1 / (1 + 10 ** ((ea - eh - HFA) / 400))
            s = 1.0 if hg[t] > ag[t] else (0.5 if hg[t] == ag[t] else 0.0)
            gd = abs(hg[t] - ag[t])
            mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
            d = K * mult * (s - exp_h)
            elo[h], elo[a] = eh + d, ea - d
            # pi-ratings
            rh, ra = pi_h.get(h, 0.0), pi_a.get(a, 0.0)
            gh = np.sign(rh) * (10 ** (abs(rh) / C) - 1)
            ga = np.sign(ra) * (10 ** (abs(ra) / C) - 1)
            err = (hg[t] - ag[t]) - (gh - ga)
            psi = C * np.log10(1 + abs(err)) * np.sign(err)
            pi_h[h] = rh + psi * LAM
            pi_a[h] = pi_a.get(h, 0.0) + psi * LAM * GAM
            pi_a[a] = ra - psi * LAM
            pi_h[a] = pi_h.get(a, 0.0) - psi * LAM * GAM
        i = j

    df = pd.DataFrame(out, index=m.index,
                      columns=["elo_home", "elo_away", "pi_home_h", "pi_away_a", "pi_home_a", "pi_away_h"])
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["elo_exp"] = 1 / (1 + 10 ** (-(df["elo_diff"] + HFA) / 400))
    pih = (df["pi_home_h"] + df["pi_home_a"]) / 2
    pia = (df["pi_away_a"] + df["pi_away_h"]) / 2
    df["pi_diff"] = df["pi_home_h"] - df["pi_away_a"]
    df["pi_diff_overall"] = pih - pia
    gh = np.sign(df["pi_home_h"]) * (10 ** (df["pi_home_h"].abs() / C) - 1)
    ga = np.sign(df["pi_away_a"]) * (10 ** (df["pi_away_a"].abs() / C) - 1)
    df["pi_exp_gd"] = gh - ga
    return df


# --------------------------------------------------------------------------------------
# Forma, tabela ligowa, odpoczynek, H2H, statystyki z XML (z poprzednich meczów)
# --------------------------------------------------------------------------------------
def _xml_count(s: pd.Series, team: pd.Series) -> pd.Series:
    """Liczba zdarzeń w XML przypisanych do drużyny (tag <team>id</team>)."""
    out = np.full(len(s), np.nan)
    for k, (x, t) in enumerate(zip(s.to_numpy(), team.to_numpy())):
        if isinstance(x, str) and x.startswith("<"):
            out[k] = len(re.findall(rf"<team>{t}</team>", x))
    return pd.Series(out, index=s.index)


def _possession(s: pd.Series) -> pd.Series:
    """Średnie posiadanie gospodarza z ostatniego wpisu <homepos>."""
    out = np.full(len(s), np.nan)
    for k, x in enumerate(s.to_numpy()):
        if isinstance(x, str):
            v = re.findall(r"<homepos>(\d+)</homepos>", x)
            if v:
                out[k] = float(v[-1])
    return pd.Series(out, index=s.index)


def form_features(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    m["date"] = pd.to_datetime(m["date"])
    m["sot_h"] = _xml_count(m["shoton"], m["home_team_api_id"])
    m["sot_a"] = _xml_count(m["shoton"], m["away_team_api_id"])
    m["pos_h"] = _possession(m["possession"])
    has_xml = m["shoton"].notna() & m["shoton"].astype(str).str.startswith("<")
    m.loc[~has_xml, ["sot_h", "sot_a"]] = np.nan
    m["pos_a"] = 100 - m["pos_h"]

    def side(is_home: bool) -> pd.DataFrame:
        p, q = ("home", "away") if is_home else ("away", "home")
        d = pd.DataFrame({
            "match": m.index, "date": m["date"], "season": m["season"], "league": m["league_id"],
            "team": m[f"{p}_team_api_id"], "opp": m[f"{q}_team_api_id"], "is_home": is_home,
            "gf": m[f"{p}_team_goal"], "ga": m[f"{q}_team_goal"],
            "sot_f": m["sot_h" if is_home else "sot_a"], "sot_a": m["sot_a" if is_home else "sot_h"],
            "pos": m["pos_h" if is_home else "pos_a"],
        })
        return d

    L = pd.concat([side(True), side(False)], ignore_index=True).sort_values(["team", "date"], kind="stable")
    L["gd"] = L["gf"] - L["ga"]
    L["pts"] = np.select([L["gf"] > L["ga"], L["gf"] == L["ga"]], [3, 1], 0)
    L["draw"] = (L["gf"] == L["ga"]).astype(float)
    g = L.groupby("team", sort=False)

    feats = {}
    for col in ["pts", "gf", "ga", "gd", "draw"]:
        prev = g[col].shift(1)
        L[f"{col}_prev"] = prev
        for w in (5, 10):
            feats[f"{col}_r{w}"] = prev.groupby(L["team"]).transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
        feats[f"{col}_ewm"] = prev.groupby(L["team"]).transform(lambda s: s.ewm(halflife=8, min_periods=1).mean())
    for col in ["sot_f", "sot_a", "pos"]:
        prev = g[col].shift(1)
        feats[f"{col}_r10"] = prev.groupby(L["team"]).transform(lambda s: s.rolling(10, min_periods=1).mean())
    # forma z meczów u siebie / na wyjeździe (venue-specific)
    gv = L.groupby(["team", "is_home"], sort=False)
    for col in ["pts", "gd"]:
        prev = gv[col].shift(1)
        feats[f"{col}_venue_r5"] = prev.groupby([L["team"], L["is_home"]]).transform(
            lambda s: s.rolling(5, min_periods=1).mean())
    feats["rest_days"] = g["date"].diff().dt.days.clip(upper=30)
    # tabela ligowa przed meczem (bieżący sezon)
    gs = L.groupby(["team", "season"], sort=False)
    feats["season_pts"] = gs["pts"].cumsum() - L["pts"]
    feats["season_gd"] = gs["gd"].cumsum() - L["gd"]
    feats["season_played"] = gs.cumcount()
    feats["season_ppg"] = feats["season_pts"] / feats["season_played"].replace(0, np.nan)
    F = pd.DataFrame(feats)
    F[["match", "is_home", "date", "season", "league", "team", "opp"]] = L[
        ["match", "is_home", "date", "season", "league", "team", "opp"]]

    # pozycja w tabeli: ranking ppg wśród drużyn ligi w danym sezonie na podstawie ostatniego znanego stanu
    # (przybliżenie: percentyl ppg w lidze/sezonie wśród meczów rozegranych w tym samym tygodniu)
    F["week"] = F["date"].dt.to_period("W")
    F["table_pct"] = F.groupby(["league", "season", "week"])["season_ppg"].rank(pct=True)

    H = F[F["is_home"]].set_index("match").drop(columns=["is_home", "date", "season", "league", "team", "opp", "week"])
    A = F[~F["is_home"]].set_index("match").drop(columns=["is_home", "date", "season", "league", "team", "opp", "week"])
    H = H.add_prefix("h_").reindex(m.index)
    A = A.add_prefix("a_").reindex(m.index)
    out = pd.concat([H, A], axis=1)
    for c in ["pts_r5", "pts_r10", "gd_r5", "gd_r10", "pts_ewm", "gd_ewm", "gf_ewm", "ga_ewm",
              "season_ppg", "sot_f_r10", "pos_r10", "rest_days", "table_pct"]:
        out[f"d_{c}"] = out[f"h_{c}"] - out[f"a_{c}"]

    # H2H: średnia różnica bramek (z perspektywy obecnego gospodarza) z poprzednich spotkań tej pary
    key = np.where(m["home_team_api_id"] < m["away_team_api_id"],
                   m["home_team_api_id"].astype(str) + "_" + m["away_team_api_id"].astype(str),
                   m["away_team_api_id"].astype(str) + "_" + m["home_team_api_id"].astype(str))
    sign = np.where(m["home_team_api_id"] < m["away_team_api_id"], 1.0, -1.0)
    hh = pd.DataFrame({"key": key, "date": m["date"], "gd": sign * (m["home_team_goal"] - m["away_team_goal"]),
                       "draw": (m["home_team_goal"] == m["away_team_goal"]).astype(float)}, index=m.index)
    hh = hh.sort_values("date", kind="stable")
    gk = hh.groupby("key")
    hh["gd_prev"] = gk["gd"].transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    hh["draw_prev"] = gk["draw"].transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    hh["n_prev"] = gk.cumcount()
    out["h2h_gd"] = (hh["gd_prev"] * pd.Series(sign, index=m.index)).reindex(m.index)
    out["h2h_draw"] = hh["draw_prev"].reindex(m.index)
    out["h2h_n"] = hh["n_prev"].reindex(m.index)
    return out


# --------------------------------------------------------------------------------------
# Składy: atrybuty FIFA (ostatni pomiar przed meczem)
# --------------------------------------------------------------------------------------
PLAYER_ATTRS = ["overall_rating", "potential"]
ROLE_ATTRS = {  # atrybuty agregowane per formacja
    "gk": ["gk_reflexes", "gk_diving", "gk_positioning", "gk_handling"],
    "def": ["standing_tackle", "marking", "interceptions"],
    "mid": ["short_passing", "vision", "ball_control"],
    "att": ["finishing", "positioning", "shot_power"],
}


def lineup_features(m: pd.DataFrame) -> pd.DataFrame:
    pa = load_table("Player_Attributes")
    pl = load_table("Player")[["player_api_id", "birthday"]]
    pa["date"] = pd.to_datetime(pa["date"])
    pa = pa.sort_values("date")
    attrs = PLAYER_ATTRS + sorted({a for v in ROLE_ATTRS.values() for a in v})

    rows = []
    for side in ("home", "away"):
        for k in range(1, 12):
            rows.append(pd.DataFrame({
                "match": m.index, "side": side, "slot": k,
                "player_api_id": m[f"{side}_player_{k}"], "Y": m[f"{side}_player_Y{k}"],
                "date": pd.to_datetime(m["date"]),
            }))
    LP = pd.concat(rows, ignore_index=True).dropna(subset=["player_api_id"])
    LP["player_api_id"] = LP["player_api_id"].astype(int)
    LP = LP.sort_values("date")
    LP = pd.merge_asof(LP, pa[["player_api_id", "date"] + attrs], on="date", by="player_api_id",
                       allow_exact_matches=False, direction="backward")
    LP = LP.merge(pl, on="player_api_id", how="left")
    LP["age"] = (LP["date"] - pd.to_datetime(LP["birthday"])).dt.days / 365.25
    # formacje po współrzędnej Y (1 = bramkarz)
    LP["role"] = np.select([LP["Y"] == 1, LP["Y"] <= 4, LP["Y"] <= 8], ["gk", "def", "mid"], "att")
    LP.loc[LP["Y"].isna(), "role"] = np.select(
        [LP["slot"] == 1, LP["slot"] <= 5, LP["slot"] <= 9], ["gk", "def", "mid"], "att")[LP["Y"].isna().to_numpy()]

    g = LP.groupby(["match", "side"])
    agg = pd.DataFrame({
        "ovr_mean": g["overall_rating"].mean(),
        "ovr_max": g["overall_rating"].max(),
        "ovr_min": g["overall_rating"].min(),
        "pot_mean": g["potential"].mean(),
        "age_mean": g["age"].mean(),
        "n_known": g["overall_rating"].count(),
    })
    for role, cols in ROLE_ATTRS.items():
        sub = LP[LP["role"] == role]
        gr = sub.groupby(["match", "side"])
        agg[f"{role}_ovr"] = gr["overall_rating"].mean()
        agg[f"{role}_skill"] = sub[cols].mean(axis=1).groupby([sub["match"], sub["side"]]).mean()
    agg = agg.unstack("side")
    agg.columns = [f"lu_{s[0]}_{c}" for c, s in agg.columns]
    agg = agg.reindex(m.index)
    for c in ["ovr_mean", "ovr_max", "pot_mean", "age_mean", "gk_ovr", "def_ovr", "mid_ovr", "att_ovr",
              "gk_skill", "def_skill", "mid_skill", "att_skill"]:
        agg[f"lu_d_{c}"] = agg[f"lu_h_{c}"] - agg[f"lu_a_{c}"]
    # atak gospodarza vs obrona gościa i odwrotnie
    agg["lu_hatt_vs_adef"] = agg["lu_h_att_skill"] - agg["lu_a_def_skill"]
    agg["lu_aatt_vs_hdef"] = agg["lu_a_att_skill"] - agg["lu_h_def_skill"]
    return agg


# --------------------------------------------------------------------------------------
# Atrybuty taktyczne drużyn
# --------------------------------------------------------------------------------------
TEAM_ATTRS = ["buildUpPlaySpeed", "buildUpPlayPassing", "chanceCreationPassing", "chanceCreationCrossing",
              "chanceCreationShooting", "defencePressure", "defenceAggression", "defenceTeamWidth"]


def team_attr_features(m: pd.DataFrame) -> pd.DataFrame:
    ta = load_table("Team_Attributes")
    ta["date"] = pd.to_datetime(ta["date"])
    ta = ta.sort_values("date")[["team_api_id", "date"] + TEAM_ATTRS]
    out = pd.DataFrame(index=m.index)
    base = pd.DataFrame({"date": pd.to_datetime(m["date"]), "match": m.index})
    for side in ("home", "away"):
        b = base.assign(team_api_id=m[f"{side}_team_api_id"]).sort_values("date")
        r = pd.merge_asof(b, ta, on="date", by="team_api_id", allow_exact_matches=False).set_index("match")
        for c in TEAM_ATTRS:
            out[f"ta_{side[0]}_{c}"] = r[c].reindex(m.index)
    for c in TEAM_ATTRS:
        out[f"ta_d_{c}"] = out[f"ta_h_{c}"] - out[f"ta_a_{c}"]
    return out


# --------------------------------------------------------------------------------------
def context_features(m: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=m.index)
    out["league_id"] = m["league_id"].astype(int)
    out["stage"] = m["stage"]
    d = pd.to_datetime(m["date"])
    out["month"] = d.dt.month
    # historyczny odsetek remisów i przewaga gospodarza w lidze (tylko poprzednie sezony)
    res = pd.DataFrame({"league": m["league_id"], "season": m["season"],
                        "draw": (m["home_team_goal"] == m["away_team_goal"]).astype(float),
                        "win": (m["home_team_goal"] > m["away_team_goal"]).astype(float)})
    per = res.groupby(["league", "season"])[["draw", "win"]].mean().sort_index()
    prev = per.groupby(level=0).transform(lambda s: s.shift(1).expanding().mean())
    j = res[["league", "season"]].merge(prev, left_on=["league", "season"], right_index=True, how="left")
    out["league_draw_rate"] = j["draw"].to_numpy()
    out["league_home_rate"] = j["win"].to_numpy()
    return out


def _poisson_probs(lh: np.ndarray, la: np.ndarray, max_goals: int = 10) -> np.ndarray:
    """P(win), P(draw), P(loss) przy niezależnych rozkładach Poissona (Maher 1982 / Dixon-Coles bez korekty)."""
    from math import factorial

    k = np.arange(max_goals + 1)
    fact = np.array([factorial(int(i)) for i in k], dtype=float)
    ph = np.exp(-lh[:, None]) * lh[:, None] ** k / fact
    pa = np.exp(-la[:, None]) * la[:, None] ** k / fact
    joint = ph[:, :, None] * pa[:, None, :]
    win = np.tril(np.ones((max_goals + 1,) * 2), -1)
    return np.c_[(joint * win).sum((1, 2)), np.einsum("nii->n", joint), (joint * win.T).sum((1, 2))]


def derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """Cechy pochodne pod remisy: bezwzględne różnice sił, oczekiwane gole, P(remis) z modelu Poissona."""
    out = pd.DataFrame(index=X.index)
    for c in ["elo_diff", "pi_diff", "pi_exp_gd", "lu_d_ovr_mean", "d_pts_ewm", "d_gd_ewm", "odds_logratio"]:
        out[f"abs_{c}"] = X[c].abs()
    # siła ataku/obrony z formy (EWM): λ_home = (atak gospodarza + obrona gościa) / 2, analogicznie λ_away
    lh = ((X["h_gf_ewm"] + X["a_ga_ewm"]) / 2).fillna(1.4).clip(0.2, 5).to_numpy()
    la = ((X["a_gf_ewm"] + X["h_ga_ewm"]) / 2).fillna(1.1).clip(0.2, 5).to_numpy()
    P = _poisson_probs(lh, la)
    out["pois_win"], out["pois_draw"], out["pois_loss"] = P[:, 0], P[:, 1], P[:, 2]
    out["exp_total_goals"] = lh + la
    out["draw_tendency"] = (X["h_draw_ewm"] + X["a_draw_ewm"]) / 2
    return out


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    key = id(matches)
    if key in _CACHE and len(_CACHE[key]) == len(matches):
        return _CACHE[key]
    X = pd.concat([
        context_features(matches),
        odds_features(matches),
        rating_features(matches),
        form_features(matches),
        lineup_features(matches),
        team_attr_features(matches),
    ], axis=1)
    X = pd.concat([X, derived_features(X)], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).astype(float)
    _CACHE[key] = X
    return X
