"""Inżynieria cech — wspólna dla obu torów.

Przetwarzanie danych i cechy są dowolne w OBU torach; ograniczenie toru CLASSIC dotyczy tylko modeli.
Wszystkie cechy liczone WYŁĄCZNIE z informacji dostępnych przed meczem (AGENTS.md, sekcja 6):
  - kursy bukmacherów (publikowane przed meczem),
  - historia wcześniejszych meczów (shift / stan "przed" bieżącym meczem),
  - atrybuty FIFA graczy i drużyn z datą ściśle wcześniejszą niż data meczu.

build_features(matches) zwraca DataFrame z tym samym indeksem co `matches`.
Wynik jest cache'owany w outputs/ (klucz = FEATURES_VERSION + liczba meczów).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import load_table

FEATURES_VERSION = "v1"
CACHE_DIR = Path("outputs")

BOOKMAKERS = ("B365", "BW", "IW", "LB", "PS", "WH", "SJ", "VC", "GB", "BS")


# ---------------------------------------------------------------------------
# Kursy
# ---------------------------------------------------------------------------
def odds_features(m: pd.DataFrame) -> pd.DataFrame:
    probs = {r: [] for r in "HDA"}
    margins = []
    for b in BOOKMAKERS:
        cols = [f"{b}{r}" for r in "HDA"]
        if not all(c in m.columns for c in cols):
            continue
        inv = 1.0 / m[cols].where(m[cols] > 1.0)
        s = inv.sum(axis=1, min_count=3)
        margins.append(s - 1)
        for r, c in zip("HDA", cols):
            probs[r].append(inv[c] / s)
    out = pd.DataFrame(index=m.index)
    for r in "HDA":
        P = pd.concat(probs[r], axis=1)
        out[f"odds_p{r}"] = P.mean(axis=1)
        out[f"odds_p{r}_std"] = P.std(axis=1)
        out[f"odds_p{r}_max"] = P.max(axis=1)
    out["odds_margin"] = pd.concat(margins, axis=1).mean(axis=1)
    out["odds_n_bookies"] = pd.concat(probs["H"], axis=1).notna().sum(axis=1)
    out["odds_pH_minus_pA"] = out["odds_pH"] - out["odds_pA"]
    out["odds_abs_diff"] = out["odds_pH_minus_pA"].abs()
    out["odds_entropy"] = -sum(out[f"odds_p{r}"] * np.log(out[f"odds_p{r}"]) for r in "HDA")
    return out


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------
def elo_features(m: pd.DataFrame, k=20.0, home_adv=65.0, season_regress=0.2) -> pd.DataFrame:
    """Elo liczone sekwencyjnie; dla meczu zapisywany jest rating PRZED meczem."""
    order = m.sort_values(["date", "match_api_id"]).index
    rating, last_season = {}, {}
    pre_h = np.empty(len(m)); pre_a = np.empty(len(m))
    pos = {ix: i for i, ix in enumerate(m.index)}
    H, A = m["home_team_api_id"], m["away_team_api_id"]
    HG, AG, S = m["home_team_goal"], m["away_team_goal"], m["season"]
    for ix in order:
        h, a, season = H[ix], A[ix], S[ix]
        for t in (h, a):
            if t not in rating:
                rating[t] = 1500.0
            elif last_season[t] != season:
                rating[t] = 1500.0 + (rating[t] - 1500.0) * (1 - season_regress)
            last_season[t] = season
        rh, ra = rating[h], rating[a]
        pre_h[pos[ix]], pre_a[pos[ix]] = rh, ra
        # aktualizacja PO zapisaniu ratingu sprzed meczu
        exp_h = 1.0 / (1.0 + 10 ** ((ra - rh - home_adv) / 400.0))
        gd = HG[ix] - AG[ix]
        res = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        mult = np.log(abs(gd) + 1) + 1.0
        delta = k * mult * (res - exp_h)
        rating[h] = rh + delta
        rating[a] = ra - delta
    out = pd.DataFrame({"elo_home": pre_h, "elo_away": pre_a}, index=m.index)
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    out["elo_exp_home"] = 1.0 / (1.0 + 10 ** (-(out["elo_diff"] + home_adv) / 400.0))
    out["elo_abs_diff"] = out["elo_diff"].abs()
    return out


# ---------------------------------------------------------------------------
# Forma / historia drużyn
# ---------------------------------------------------------------------------
def _long(m: pd.DataFrame) -> pd.DataFrame:
    base = m[["date", "season", "league_id", "match_api_id"]]
    home = base.assign(
        team=m["home_team_api_id"], opp=m["away_team_api_id"], is_home=1,
        gf=m["home_team_goal"], ga=m["away_team_goal"], midx=m.index,
    )
    away = base.assign(
        team=m["away_team_api_id"], opp=m["home_team_api_id"], is_home=0,
        gf=m["away_team_goal"], ga=m["home_team_goal"], midx=m.index,
    )
    lg = pd.concat([home, away], ignore_index=True)
    lg["pts"] = np.select([lg.gf > lg.ga, lg.gf == lg.ga], [3, 1], 0)
    lg["win"] = (lg.gf > lg.ga).astype(float)
    lg["draw"] = (lg.gf == lg.ga).astype(float)
    lg["gd"] = lg.gf - lg.ga
    return lg.sort_values(["team", "date", "match_api_id"]).reset_index(drop=True)


def _roll(g, col, n):
    return g[col].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())


def form_features(m: pd.DataFrame) -> pd.DataFrame:
    lg = _long(m)
    g = lg.groupby("team", sort=False)
    feats = {}
    for n in (3, 5, 10, 20):
        for col in ("pts", "gf", "ga"):
            feats[f"form{n}_{col}"] = _roll(g, col, n)
    for n in (10,):
        feats[f"form{n}_draw"] = _roll(g, "draw", n)
    # forma wg miejsca (dom / wyjazd)
    gv = lg.groupby(["team", "is_home"], sort=False)
    for n in (5, 10):
        for col in ("pts", "gf", "ga"):
            feats[f"venue{n}_{col}"] = _roll(gv, col, n)
    # sezon do tej pory
    gs = lg.groupby(["team", "season"], sort=False)
    played = gs.cumcount()
    feats["season_played"] = played
    for col in ("pts", "gf", "ga", "draw"):
        feats[f"season_{col}_pg"] = (gs[col].cumsum() - lg[col]) / played.replace(0, np.nan)
    # odpoczynek
    feats["rest_days"] = g["date"].diff().dt.days.clip(upper=30)
    # mecze bezpośrednie (z perspektywy drużyny)
    gh = lg.groupby(["team", "opp"], sort=False)
    feats["h2h_pts"] = _roll(gh, "pts", 5)
    feats["h2h_gd"] = _roll(gh, "gd", 5)
    feats["h2h_n"] = gh.cumcount().clip(upper=10)
    F = pd.DataFrame(feats)
    F["midx"], F["is_home"] = lg["midx"], lg["is_home"]
    home = F[F.is_home == 1].set_index("midx").drop(columns="is_home").add_prefix("h_")
    away = F[F.is_home == 0].set_index("midx").drop(columns="is_home").add_prefix("a_")
    out = home.join(away).reindex(m.index)
    for c in feats:
        if c.startswith(("h2h_n",)):
            continue
        out[f"d_{c}"] = out[f"h_{c}"] - out[f"a_{c}"]
    out = out.drop(columns=["a_h2h_n"])  # symetryczne
    return out


def table_features(m: pd.DataFrame) -> pd.DataFrame:
    """Pozycja w tabeli ligi (punkty, różnica bramek) PRZED meczem."""
    out = np.full((len(m), 4), np.nan)
    pos = {ix: i for i, ix in enumerate(m.index)}
    for _, d in m.sort_values(["date", "match_api_id"]).groupby(["league_id", "season"], sort=False):
        teams = pd.unique(d[["home_team_api_id", "away_team_api_id"]].values.ravel())
        pts = dict.fromkeys(teams, 0); gd = dict.fromkeys(teams, 0)
        n = len(teams)
        # mecze z tego samego dnia: tabela sprzed dnia
        for date, day in d.groupby("date", sort=True):
            ranking = sorted(teams, key=lambda t: (-pts[t], -gd[t]))
            rank = {t: i + 1 for i, t in enumerate(ranking)}
            for ix, r in day.iterrows():
                h, a = r.home_team_api_id, r.away_team_api_id
                out[pos[ix]] = [rank[h] / n, rank[a] / n, pts[h] - pts[a], gd[h] - gd[a]]
            for _, r in day.iterrows():
                h, a, hg, ag = r.home_team_api_id, r.away_team_api_id, r.home_team_goal, r.away_team_goal
                gd[h] += hg - ag; gd[a] += ag - hg
                if hg > ag: pts[h] += 3
                elif hg < ag: pts[a] += 3
                else: pts[h] += 1; pts[a] += 1
    return pd.DataFrame(out, index=m.index, columns=["tbl_rank_h", "tbl_rank_a", "tbl_pts_diff", "tbl_gd_diff"])


# ---------------------------------------------------------------------------
# Składy + atrybuty FIFA
# ---------------------------------------------------------------------------
PLAYER_ATTRS = ["overall_rating", "potential"]


def lineup_features(m: pd.DataFrame) -> pd.DataFrame:
    pa = load_table("Player_Attributes")
    pa["date"] = pd.to_datetime(pa["date"])
    pa = pa.dropna(subset=["overall_rating"]).sort_values("date")
    pl = load_table("Player")[["player_api_id", "birthday", "height"]]
    pl["birthday"] = pd.to_datetime(pl["birthday"])

    rows = []
    for side in ("home", "away"):
        for i in range(1, 12):
            rows.append(pd.DataFrame({
                "midx": m.index, "date": m["date"], "side": side, "slot": i,
                "player_api_id": m[f"{side}_player_{i}"], "Y": m[f"{side}_player_Y{i}"],
            }))
    L = pd.concat(rows, ignore_index=True).dropna(subset=["player_api_id"])
    L["player_api_id"] = L["player_api_id"].astype("int64")
    L = L.sort_values("date")
    L = pd.merge_asof(L, pa[["player_api_id", "date", *PLAYER_ATTRS]], on="date",
                      by="player_api_id", allow_exact_matches=False, direction="backward")
    L = L.merge(pl, on="player_api_id", how="left")
    L["age"] = (L["date"] - L["birthday"]).dt.days / 365.25
    L["zone"] = pd.cut(L["Y"], bins=[0, 1.5, 4.5, 8.5, 20], labels=["gk", "def", "mid", "att"])

    agg = L.groupby(["midx", "side"]).agg(
        ovr_mean=("overall_rating", "mean"), ovr_max=("overall_rating", "max"),
        ovr_min=("overall_rating", "min"), ovr_std=("overall_rating", "std"),
        pot_mean=("potential", "mean"), age_mean=("age", "mean"), height_mean=("height", "mean"),
        n_players=("player_api_id", "size"), n_rated=("overall_rating", "count"),
    )
    zone = L.groupby(["midx", "side", "zone"], observed=True)["overall_rating"].mean().unstack("zone")
    zone.columns = [f"ovr_{z}" for z in zone.columns]
    agg = agg.join(zone).unstack("side")
    agg.columns = [f"{'h' if s == 'home' else 'a'}_{c}" for c, s in agg.columns]
    out = agg.reindex(m.index)
    for c in ["ovr_mean", "ovr_max", "ovr_min", "pot_mean", "age_mean", "ovr_gk", "ovr_def", "ovr_mid", "ovr_att"]:
        if f"h_{c}" in out and f"a_{c}" in out:
            out[f"d_{c}"] = out[f"h_{c}"] - out[f"a_{c}"]
    return out


TEAM_ATTRS = ["buildUpPlaySpeed", "buildUpPlayPassing", "chanceCreationPassing", "chanceCreationCrossing",
              "chanceCreationShooting", "defencePressure", "defenceAggression", "defenceTeamWidth"]


def team_attr_features(m: pd.DataFrame) -> pd.DataFrame:
    ta = load_table("Team_Attributes")
    ta["date"] = pd.to_datetime(ta["date"])
    ta = ta.sort_values("date")[["team_api_id", "date", *TEAM_ATTRS]]
    out = pd.DataFrame(index=m.index)
    for side, p in (("home", "h"), ("away", "a")):
        q = pd.DataFrame({"midx": m.index, "date": m["date"], "team_api_id": m[f"{side}_team_api_id"]}).sort_values("date")
        q = pd.merge_asof(q, ta, on="date", by="team_api_id", allow_exact_matches=False).set_index("midx")
        for c in TEAM_ATTRS:
            out[f"{p}_ta_{c}"] = q[c]
    for c in TEAM_ATTRS:
        out[f"d_ta_{c}"] = out[f"h_ta_{c}"] - out[f"a_ta_{c}"]
    return out


# ---------------------------------------------------------------------------
def context_features(m: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=m.index)
    out["stage"] = m["stage"]
    out["month"] = m["date"].dt.month
    for lid in sorted(m["league_id"].unique()):
        out[f"league_{lid}"] = (m["league_id"] == lid).astype(float)
    return out


def _build(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    m["date"] = pd.to_datetime(m["date"])
    parts = [
        odds_features(m), elo_features(m), form_features(m), table_features(m),
        lineup_features(m), team_attr_features(m), context_features(m),
    ]
    X = pd.concat(parts, axis=1)
    return X.astype("float64")


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    CACHE_DIR.mkdir(exist_ok=True)
    key = f"{FEATURES_VERSION}_{len(matches)}_{int(matches['match_api_id'].sum()) % 10**9}"
    path = CACHE_DIR / f"features_{key}.pkl"
    if path.exists():
        X = pickle.loads(path.read_bytes())
    else:
        X = _build(matches)
        path.write_bytes(pickle.dumps(X))
    return X.loc[matches.index]
