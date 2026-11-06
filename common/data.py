"""Wspólny loader danych konkursowych (European Soccer Database z Kaggle: hugomathien/soccer).

NIE MODYFIKOWAĆ na własnych branchach — wszyscy muszą wczytywać identyczne dane.

Zbiór (~300 MB, plik SQLite `database.sqlite`) jest pobierany przez kagglehub przy pierwszym
wywołaniu i cache'owany w ~/.cache/kagglehub — kolejne wywołania nie pobierają go ponownie.
Publiczny zbiór — logowanie do Kaggle nie jest wymagane.

Uwaga: celowo używamy `kagglehub.dataset_download()` + `pandas.read_sql_query()` zamiast
`kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, ..., "database.sqlite", sql_query=...)`.
Wariant z adapterem pobiera pojedynczy plik i w kagglehub ma błąd wznawiania pobierania
(myli rozmiar skompresowany z rozpakowanym), a sama funkcja jest oznaczona jako deprecated.
Wynik (DataFrame z zapytania SQL) jest identyczny.

Użycie:
    from common.data import load_table, load_query, TABLES
    matches = load_table("Match")
    df = load_query("SELECT * FROM Match WHERE season = '2015/2016'")
"""

import sqlite3
from functools import cache
from pathlib import Path

import kagglehub
import pandas as pd

DATASET = "hugomathien/soccer"
FILE_NAME = "database.sqlite"

TABLES = (
    "Country",
    "League",
    "Match",
    "Player",
    "Player_Attributes",
    "Team",
    "Team_Attributes",
)


@cache
def db_path() -> Path:
    """Ścieżka do lokalnej (cache) kopii bazy; pobiera zbiór, jeśli go jeszcze nie ma."""
    path = Path(kagglehub.dataset_download(DATASET)) / FILE_NAME
    if not path.is_file():  # uszkodzony/niepełny cache → pobierz ponownie
        path = Path(kagglehub.dataset_download(DATASET, force_download=True)) / FILE_NAME
    return path


def load_query(sql_query: str, **pandas_kwargs) -> pd.DataFrame:
    """Wykonuje zapytanie SQL na surowej bazie (tylko do odczytu) i zwraca DataFrame."""
    uri = f"{db_path().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        return pd.read_sql_query(sql_query, con, **pandas_kwargs)


def load_table(name: str) -> pd.DataFrame:
    """Wczytuje całą surową tabelę bez żadnego przetwarzania."""
    if name not in TABLES:
        raise ValueError(f"Nieznana tabela {name!r}. Dostępne: {', '.join(TABLES)}")
    return load_query(f'SELECT * FROM "{name}"')


def load_all() -> dict[str, pd.DataFrame]:
    """Wczytuje wszystkie tabele jako słownik {nazwa: DataFrame}."""
    return {name: load_table(name) for name in TABLES}


if __name__ == "__main__":
    print(f"Baza: {db_path()}")
    for name, df in load_all().items():
        print(f"{name:<18} {df.shape}")
