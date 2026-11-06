"""Inżynieria cech — wspólna dla obu torów (możesz też mieć osobne cechy dla każdego toru).

Przetwarzanie danych i cechy są dowolne w OBU torach; ograniczenie toru CLASSIC dotyczy tylko modeli.
Pamiętaj: wyłącznie informacje dostępne PRZED meczem (AGENTS.md, sekcja 6).
"""

import pandas as pd

BOOKMAKERS = ("B365", "BW", "IW", "LB", "WH", "VC")


def build_raw_odds_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Baseline: implikowane prawdopodobieństwa z kursów bukmacherów."""
    odds_cols = [f"{b}{r}" for b in BOOKMAKERS for r in "HDA" if f"{b}{r}" in matches.columns]
    return 1.0 / matches[odds_cols].where(matches[odds_cols] > 0)


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Cechy kursowe z surowymi i znormalizowanymi prawdopodobieństwami przedmeczowymi."""
    features = build_raw_odds_features(matches)
    normalized = {}

    for bookmaker in BOOKMAKERS:
        cols = [f"{bookmaker}{result}" for result in "HDA"]
        if not all(col in matches.columns for col in cols):
            continue

        implied = features[cols]
        overround = implied.sum(axis=1, min_count=3)
        probabilities = implied.div(overround, axis=0)
        for result, col in zip("HDA", cols):
            name = f"{bookmaker}_{result}_normalized"
            features[name] = probabilities[col]
            normalized.setdefault(result, []).append(probabilities[col].rename(name))
        features[f"{bookmaker}_overround"] = overround

    for result, series in normalized.items():
        probabilities = pd.concat(series, axis=1)
        features[f"consensus_{result}"] = probabilities.mean(axis=1)
        features[f"disagreement_{result}"] = probabilities.std(axis=1, ddof=0)

    features["available_bookmakers"] = sum(
        features[f"{bookmaker}_overround"].notna()
        for bookmaker in BOOKMAKERS
        if f"{bookmaker}_overround" in features
    )
    return features
