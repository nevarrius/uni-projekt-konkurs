"""Inżynieria cech — wspólna dla obu torów (możesz też mieć osobne cechy dla każdego toru).

Przetwarzanie danych i cechy są dowolne w OBU torach; ograniczenie toru CLASSIC dotyczy tylko modeli.
Pamiętaj: wyłącznie informacje dostępne PRZED meczem (AGENTS.md, sekcja 6).
"""

import pandas as pd

BOOKMAKERS = ("B365", "BW", "IW", "LB", "WH", "VC")


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Baseline: implikowane prawdopodobieństwa z kursów bukmacherów (publikowane przed meczem)."""
    odds_cols = [f"{b}{r}" for b in BOOKMAKERS for r in "HDA" if f"{b}{r}" in matches.columns]
    return 1.0 / matches[odds_cols]
