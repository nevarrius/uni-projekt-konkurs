# Konkurs agentów AI — instrukcja dla uczestników i ich agentów

> **Jeśli jesteś agentem AI: przeczytaj ten plik w całości, zanim cokolwiek zrobisz.**
> Ten plik jest wspólnym „regulaminem” dla wszystkich branchy. Nie modyfikuj go na swoim branchu
> (zmiany regulaminu uzgadnia zespół i trafiają one do `main` przez organizatora).

---

## 1. Cel

Na podstawie **nieprzetworzonego (surowego) zbioru danych** zbudować kod, który osiąga
**możliwie najlepszy wynik predykcji** w metryce opisanej w sekcji 2 — **w dwóch torach naraz**
(sekcja 3): **DOWOLNYM** (`open`) i **KLASYCZNYM** (`classic`, tylko wskazane modele).
W obu torach chcemy uzyskać możliwie najlepszy wynik.

Każdy uczestnik (człowiek + jego agent) pracuje na **własnym branchu**. Branche są publiczne
w repozytorium, więc wszyscy mogą na bieżąco podglądać wyniki i pomysły innych, czerpać z nich
i je ulepszać. Zwycięża najlepsze rozwiązanie spośród wszystkich branchy — konkurujemy,
ale też wspólnie podnosimy poprzeczkę.

---

## 2. Zadanie

> **Polecenie:** stworzyć klasyfikator przypisujący mecz do jednej z 3 klas decyzyjnych —
> **wygrana, przegrana, remis**. Celem jest uzyskanie jak najwyższej **zbalansowanej dokładności (BA)**.
> Predykcja jest wykonywana na danych czasowych, więc do trenowania wykorzystujemy **starsze mecze**,
> a do testowania **nowsze**.

| Element | Wartość |
|---|---|
| Typ zadania | klasyfikacja wieloklasowa (3 klasy), jeden wiersz = jeden mecz z tabeli `Match` |
| Zbiór danych | Kaggle [`hugomathien/soccer`](https://www.kaggle.com/datasets/hugomathien/soccer) — European Soccer Database, plik SQLite ~300 MB. Wczytywanie **wyłącznie** przez `common/data.py` (sekcja 2.1) |
| Zmienna celu | wynik z perspektywy **gospodarza**: `win` (home_team_goal > away_team_goal), `draw`, `loss` — tworzona **wyłącznie** przez `common.evaluation.make_target` |
| Rozkład klas | win ≈ 46%, loss ≈ 29%, draw ≈ 25% (stabilny między sezonami) — klasy niezbalansowane, remis najtrudniejszy |
| **Metryka główna** | **balanced accuracy** (`sklearn.metrics.balanced_accuracy_score`), wyższa = lepsza. Losowe zgadywanie = 0.333 |
| Podział czasowy | trening na starszych sezonach, walidacja/test na nowszych — szczegóły w sekcji 6 |
| Tory | **dwa osobne rankingi**: `open` (dowolne metody) i `classic` (tylko lista modeli) — sekcja 3 |
| Wynik rankingowy | `cv_score` (walk-forward CV, sezony 2012/13–2014/15); **ostateczny werdykt: `holdout_score`** na sezonie 2015/2016 — w każdym torze osobno |
| Baseline `classic` | LogReg na kursach bukmacherów: **CV BA = 0.4585 ± 0.013, holdout 0.4565** |
| Baseline `open` | HistGradientBoosting na kursach: **CV BA = 0.4535 ± 0.010, holdout 0.4509** (tor `open` może też używać modeli klasycznych, więc realny próg to 0.4585) |

### 2.1. Dane

Dane **nie są w repozytorium** — pobiera je kod przez `kagglehub` przy pierwszym uruchomieniu
(cache w `~/.cache/kagglehub`, kolejne uruchomienia nie pobierają ponownie). Zbiór jest publiczny,
logowanie do Kaggle nie jest wymagane.

Wspólny loader `common/data.py` (**nie modyfikuj go na swoim branchu** — wszyscy muszą startować
z identycznych surowych danych; całe przetwarzanie robisz we własnym kodzie):

```python
from common.data import load_table, load_query, load_all, TABLES

matches = load_table("Match")                      # cała surowa tabela
df = load_query("SELECT * FROM Match WHERE season = '2015/2016'")   # dowolne SQL
```

Szybki test: `uv run python -m common.data` (wypisuje wymiary wszystkich tabel).

| Tabela | Wiersze × kolumny | Zawartość |
|---|---|---|
| `Match` | 25 979 × 115 | mecze 2008–2016, 11 lig: wynik, składy (ID i pozycje X/Y graczy), statystyki zdarzeń (XML), kursy bukmacherów |
| `Player_Attributes` | 183 978 × 42 | atrybuty graczy z FIFA, **z datą** (wiele pomiarów na gracza) |
| `Player` | 11 060 × 7 | dane graczy (urodziny, wzrost, waga) |
| `Team_Attributes` | 1 458 × 25 | atrybuty taktyczne drużyn z FIFA, **z datą** |
| `Team` | 299 × 5 | nazwy drużyn |
| `League`, `Country` | 11 × 3, 11 × 2 | słowniki lig i krajów |

Klucze łączenia: `*_api_id` (np. `Match.home_team_api_id` → `Team.team_api_id`,
`Match.home_player_1` → `Player.player_api_id`). Surowe dane zawierają dużo braków
(szczególnie składy, kursy i kolumny XML) — to część zadania.

---

## 3. Dwa tory konkursu

Konkurs ma **dwa niezależne rankingi**. **Każdy branch rozwija oba tory równolegle** i w obu
walczy o możliwie najlepszy wynik. Zadanie, dane, target, metryka i protokół ewaluacji są w obu
torach **identyczne** — różnią się wyłącznie dozwolone modele.

| Tor | Plik rozwiązania | Dozwolone modele |
|---|---|---|
| **DOWOLNY** (`open`) | `solutions/open.py` | **dowolne** — gradient boosting (LightGBM/XGBoost/CatBoost), stacking, własne sieci, ensembling czegokolwiek, także modele z toru klasycznego |
| **KLASYCZNY** (`classic`) | `solutions/classic.py` | **wyłącznie:** regresja logistyczna, SVM, KNN, drzewa decyzyjne, Bagging, AdaBoost, Random Forest, Ensemble voting, sieci neuronowe |

### 3.1. Zasady toru KLASYCZNEGO

Ograniczenie dotyczy **modeli**, czyli wszystkiego, co uczy się przewidywać. **Przetwarzanie danych
i inżynieria cech są dowolne** w obu torach: pandas, imputacja, skalowanie, PCA, encodery,
ręcznie liczone cechy (Elo, forma, agregaty). Wolno też dobierać progi i wagi klas na
prawdopodobieństwach z dozwolonego modelu.

Mapowanie na klasy (sprawdzane automatycznie przez `common.tracks.assert_classic`):

| Kategoria | Dozwolone klasy |
|---|---|
| Regresja logistyczna | `LogisticRegression`, `LogisticRegressionCV` |
| SVM | `SVC`, `LinearSVC`, `NuSVC` |
| KNN | `KNeighborsClassifier`, `RadiusNeighborsClassifier` |
| Drzewa decyzyjne | `DecisionTreeClassifier` |
| Bagging | `BaggingClassifier` (z dozwolonym modelem bazowym) |
| AdaBoost | `AdaBoostClassifier` (z dozwolonym modelem bazowym) |
| Random Forest | `RandomForestClassifier` |
| Ensemble voting | `VotingClassifier` (hard/soft, tylko dozwolone modele w środku) |
| Sieci neuronowe | `MLPClassifier`; dowolne sieci w PyTorch / skorch / Keras |
| Pomocnicze | `CalibratedClassifierCV` (kalibracja dozwolonego modelu), `Pipeline`, `GridSearchCV` itp. |

Wersje regresyjne tych modeli (np. `RandomForestRegressor`) też są dozwolone, np. do imputacji.

**Niedozwolone w torze klasycznym** (przykłady): `GradientBoosting*`, `HistGradientBoosting*`,
XGBoost, LightGBM, CatBoost, `ExtraTreesClassifier`, `StackingClassifier`, Naive Bayes, LDA/QDA,
`SGDClassifier`, `RidgeClassifier`. Zabronione jest też ukrywanie ich w przetwarzaniu, np.
`SelectFromModel(LGBMClassifier())` albo cechy będące predykcjami niedozwolonego modelu.

**Obowiązkowo** wywołaj `assert_classic(model)` na każdym modelu toru klasycznego przed `fit`.
Funkcja rzuca wyjątek, jeśli w (zagnieżdżonym) estymatorze jest coś spoza listy.
W razie wątpliwości co do modelu — zapytaj zespół, zamiast zakładać, że wolno.

### 3.2. Wspólny kod między torami

- Cechy mogą być wspólne (`solutions/features.py`) albo osobne dla każdego toru.
- Pomysły na cechy z toru `open` wolno przenosić do `classic` i odwrotnie.
- Tor `open` jest nadzbiorem toru `classic`, więc jego wynik nie powinien być gorszy.
  Jeśli jest gorszy, to znak, że warto przenieść tam najlepszy model klasyczny lub z nim ensemblować.

---

## 4. Zasady pracy z gitem (OBOWIĄZKOWE)

1. **Nigdy nie pushuj do `main`.** Nie twórz też PR-ów do `main`, chyba że zespół wprost o to poprosi.
2. Pracujesz wyłącznie na swoim branchu o nazwie:
   ```
   agent/<imie-lub-nick>
   ```
   np. `agent/wojtek`. Utworzenie:
   ```bash
   git checkout main && git pull
   git checkout -b agent/<nick>
   git push -u origin agent/<nick>
   ```
3. **Commituj i pushuj często** — co najmniej po każdym eksperymencie, który poprawił wynik,
   oraz regularnie w trakcie pracy (cel: nie rzadziej niż co ~30–60 min aktywnej pracy).
   Inni porównują się z Twoim branchem w czasie rzeczywistym.
4. **Nie modyfikuj cudzych branchy** — żadnych pushy, `--force` ani usuwania cudzych branchy.
   Na własnym branchu również unikaj `push --force` (historia wyników ma być czytelna).
5. Konwencja wiadomości commitów:
   ```
   [open cv=0.4731±0.010] LightGBM + forma z 5 ostatnich meczów
   [classic cv=0.4690±0.012] soft voting LogReg+RF+MLP
   [wip] eksploracja brakujących wartości
   ```
   Prefiks `[<tor> cv=...]` stosuj, gdy commit zmienia wynik danego toru; `[wip]` dla pracy w toku.
6. Dane, modele i duże artefakty **nie trafiają do repo** — patrz `.gitignore`.

---

## 5. Wymagana struktura brancha

Każdy branch musi zawierać (na swoim poziomie głównym):

```
AGENTS.md            # ten plik (bez zmian)
SCORE.json           # aktualny najlepszy wynik — format niżej (OBOWIĄZKOWY)
EXPERIMENTS.md       # dziennik eksperymentów (OBOWIĄZKOWY)
main.py              # punkt wejścia: pobiera dane, liczy OBA tory, zapisuje SCORE.json
pyproject.toml       # zależności (zarządzane przez uv) + uv.lock
solutions/open.py    # tor DOWOLNY:   build_features, fit_predict, MODELS, SUMMARY
solutions/classic.py # tor KLASYCZNY: build_features, fit_predict, MODELS, SUMMARY
solutions/...        # dowolny własny kod (cechy, modele, narzędzia)
common/data.py       # wspólny loader danych (bez zmian)
common/evaluation.py # wspólny target, podział czasowy i metryka (bez zmian)
common/tracks.py     # definicja torów + walidator assert_classic (bez zmian)
scripts/leaderboard.py  # ranking branchy (bez zmian)
```

Uruchomienie: `uv run python main.py` (oba tory) lub `uv run python main.py --track classic`
(jeden tor — wynik drugiego zostaje w `SCORE.json` bez zmian).

Interfejs modułu toru: `build_features(matches) -> X` (jeden wiersz na mecz, ten sam indeks co
`matches`), `fit_predict(X_train, y_train, X_test) -> y_pred`, `MODELS` (lista nazw użytych
modeli — deklaracja dla innych), `SUMMARY` (krótki opis podejścia).

### 5.1. `SCORE.json`

Generowany przez `main.py` przy każdym uruchomieniu (nie edytuj go ręcznie) i commitowany,
gdy wynik się poprawił. Ustaw `AUTHOR` i `AGENT` na górze `main.py`. Format (czyta go skrypt rankingowy):

```json
{
  "author": "wojtek",
  "agent": "claude-opus-5-5",
  "metric": "balanced_accuracy",
  "higher_is_better": true,
  "tracks": {
    "open": {
      "cv_score": 0.4731, "cv_std": 0.0102, "holdout_score": 0.4688,
      "folds": {"2012/2013": 0.4612, "2013/2014": 0.4755, "2014/2015": 0.4826},
      "models": ["LGBMClassifier", "LogisticRegression"],
      "summary": "LightGBM + LogReg blend, forma z 5 meczów, Elo, kursy, atrybuty FIFA składów",
      "updated": "2026-11-06T14:30:00"
    },
    "classic": {
      "cv_score": 0.4690, "cv_std": 0.0120, "holdout_score": 0.4655,
      "folds": {"2012/2013": 0.4570, "2013/2014": 0.4702, "2014/2015": 0.4798},
      "models": ["VotingClassifier", "LogisticRegression", "RandomForestClassifier", "MLPClassifier"],
      "summary": "soft voting LogReg+RF+MLP na tych samych cechach",
      "updated": "2026-11-06T14:10:00"
    }
  }
}
```

- Pola wyników pochodzą z `common.evaluation.evaluate(...)` — bez ręcznych poprawek.
- Wpisuj **tylko wynik, który `main.py` faktycznie odtwarza** na aktualnym commicie (dla obu torów).

### 5.2. `EXPERIMENTS.md`

Dziennik tylko do dopisywania (append-only), najnowsze na górze, każdy wpis oznaczony torem
(`[open]`, `[classic]` lub `[oba]`). Wpisuj także **porażki** —
to cenna informacja dla innych („to nie działa, nie traćcie czasu”).

```markdown
## 2026-11-06 14:30 — [open] forma drużyn z 5 ostatnich meczów  →  cv=0.4731 ± 0.0102  (poprawa +0.012)
- Co: średnie punkty/gole strzelone/stracone z 5 poprzednich meczów gospodarza i gościa (shift — bez bieżącego meczu).
- Dlaczego: same kursy słabo rozróżniają remisy.
- Wynik: poprawa, recall remisu 0.18 → 0.24. Czas 40 s.
- Inspiracja: agent/ania @ 3f2c1ab (pomysł z różnicą Elo).
```

---

## 6. Protokół ewaluacji (żeby wyniki były porównywalne)

Wyniki z różnych branchy są porównywalne **tylko wtedy**, gdy wszyscy liczą je tak samo,
dlatego target, podział i metryka są zaimplementowane we wspólnym `common/evaluation.py`.
**Zawsze licz wynik przez `evaluate(...)`**, nie przez własny kod:

```python
from common.evaluation import make_target, evaluate

# matches: surowa tabela Match (wszystkie 25 979 meczów), X: Twoje cechy, ten sam indeks
y = make_target(matches)
result = evaluate(fit_predict, X, y, matches["season"])   # fit_predict(X_tr, y_tr, X_te) -> y_pred
```

**Podział czasowy po sezonach** (sezony nie zachodzą na siebie datami):

| | Trening | Walidacja / test |
|---|---|---|
| CV fold 1 | 2008/09 – 2011/12 | 2012/2013 |
| CV fold 2 | 2008/09 – 2012/13 | 2013/2014 |
| CV fold 3 | 2008/09 – 2013/14 | 2014/2015 |
| **Holdout (test)** | 2008/09 – 2014/15 | **2015/2016** |

- `cv_score` = średnia BA z 3 foldów walk-forward — **na nim podejmujesz decyzje** (cechy,
  hiperparametry, wybór modelu, próg/wagi klas) i **według niego liczony jest ranking**.
- `holdout_score` (sezon 2015/2016) to **ostateczny wynik konkursu**. **Nie wolno używać go do
  żadnych decyzji** — liczysz go przy okazji, raportujesz, ale nie stroisz pod niego.
  Strojenie pod holdout i podglądanie go w pętli = przeuczenie do testu i wynik, którego nikt nie powtórzy.
- **Nie usuwaj meczów** z sezonów walidacyjnych/testowych (np. z brakami) — model musi zwrócić
  predykcję dla każdego meczu. Braki obsłuż imputacją / wartościami domyślnymi.
  Usuwanie wierszy ze zbioru **treningowego** jest dozwolone (robisz to wewnątrz `fit_predict`).
- `fit_predict` za każdym razem trenuje model **od zera** tylko na przekazanym `X_train`.
  Tuning hiperparametrów wewnątrz `fit_predict` może używać wyłącznie danych treningowych
  (np. wewnętrzny walk-forward na ostatnim sezonie treningowym).
- Optymalizujemy **balanced accuracy**, więc przydają się: `class_weight="balanced"`,
  dostrajanie progów/wag decyzji dla klas (zwłaszcza remisu), kalibracja prawdopodobieństw.
  Wszystkie takie decyzje też tylko na danych treningowych.
- **Zakaz wycieku danych (data leakage):** wszystkie transformacje uczone na danych
  (skalowanie, imputacja, encodery, selekcja cech, target encoding) dopasowuj **wyłącznie
  na części treningowej danego folda** — najlepiej przez `sklearn.pipeline.Pipeline`.
- **Wyciek czasowy (najważniejsza pułapka w tym zbiorze):** cechy meczu wolno budować wyłącznie
  z informacji dostępnych **przed jego rozpoczęciem**:
  - ✅ kursy bukmacherów (`B365H` itd. — publikowane przed meczem), składy wyjściowe i ich pozycje,
    ostatni pomiar `Player_Attributes` / `Team_Attributes` z `date` **<** data meczu,
    statystyki z **poprzednich** meczów (forma, Elo, bilans bezpośrednich spotkań) — liczone z `shift`,
    bez bieżącego meczu. Takie cechy historyczne mogą korzystać z wcześniejszych meczów
    również z sezonu walidacyjnego/testowego (w realnym użyciu też byłyby znane).
  - ❌ `home_team_goal`, `away_team_goal` i kolumny XML z przebiegu meczu (`goal`, `shoton`,
    `shotoff`, `foulcommit`, `card`, `cross`, `corner`, `possession`) **bieżącego** meczu.
    Wolno ich używać tylko w agregatach z meczów wcześniejszych.
  - ❌ atrybuty FIFA z datą późniejszą niż mecz, statystyki „całego sezonu” obejmujące mecze przyszłe.
- Wynik musi być **odtwarzalny**: `uv run python main.py` na czystym checkoucie
  (dane pobierane automatycznie) daje wynik z `SCORE.json` (z tolerancją losowości ~±0.001).
- Ustal ziarna losowości (`numpy`, `random`, modele) na 42.

Wynik podejrzanie dobry? Najpierw szukaj wycieku, potem świętuj.

---

## 7. Środowisko

- Python **3.13**, menedżer pakietów **uv** (nie używamy `pip`).
- Instalacja: `uv sync`; dodanie biblioteki: `uv add <pakiet>` (commituj `pyproject.toml` i `uv.lock`).
- Bazowo zainstalowane: `kagglehub[pandas-datasets]` (z `pandas`) i `scikit-learn`. Resztę
  (`lightgbm`, `xgboost`, `catboost`, `optuna` itd.) dodaj sam przez `uv add`.
- Uruchomienie: `uv run python main.py`.

---

## 8. Porównywanie się z innymi branchami

Ranking wszystkich branchy:

```bash
uv run python scripts/leaderboard.py
```

Skrypt robi `git fetch`, czyta `SCORE.json` z każdego zdalnego brancha `agent/*` i drukuje
**osobną tabelę dla każdego toru** (`--by holdout_score` sortuje po wyniku końcowym).

Podgląd cudzego rozwiązania bez przełączania brancha:

```bash
git fetch --all --prune
git log --oneline origin/agent/<nick> -10
git show origin/agent/<nick>:EXPERIMENTS.md
git diff agent/<moj-nick> origin/agent/<nick> -- solutions/
```

Przeniesienie konkretnego pomysłu: `git cherry-pick <commit>` lub ręczne przepisanie.
**Zawsze odnotuj źródło** w `EXPERIMENTS.md` (branch + hash commita).
Kopiowanie i ulepszanie cudzych pomysłów jest **dozwolone i pożądane** — o to chodzi w tym konkursie.

---

## 9. Pętla pracy agenta

1. **Start:** przeczytaj ten plik → utwórz/przełącz się na `agent/<nick>` → `uv sync` →
   `uv run python main.py` (pobierze dane i odtworzy baseline obu torów: classic ≈ 0.4585, open ≈ 0.4535).
2. **Rozpoznanie konkurencji:** uruchom `scripts/leaderboard.py`, przeczytaj `EXPERIMENTS.md`
   z 2–3 najlepszych branchy **w każdym torze**. Nie powtarzaj eksperymentów, które innym już nie wyszły.
3. **Pierwszy wynik:** ustaw `AUTHOR`/`AGENT` w `main.py`, wygeneruj `SCORE.json`,
   załóż `EXPERIMENTS.md`, commit + push. Działający słaby wynik jest lepszy niż brak wyniku.
4. **Iteracja** (powtarzaj, **naprzemiennie w obu torach** — żaden tor nie może zostać zaniedbany):
   1. Postaw jedną hipotezę (np. nowa cecha, inny model, tuning, ensembling).
   2. Zaimplementuj, oceń protokołem z sekcji 6 (`--track <tor>` dla szybkości).
   3. Dopisz wpis do `EXPERIMENTS.md` (również przy porażce).
   4. Jeśli lepiej → `SCORE.json` aktualizuje się sam, commit `[<tor> cv=...]`, **push**.
   5. Nowa cecha pomogła w jednym torze? Sprawdź ją od razu w drugim.
   6. Co kilka iteracji wróć do kroku 2 — sprawdź, czy ktoś Cię nie wyprzedził i czego można się od niego nauczyć.
5. **Raportuj użytkownikowi** postępy krótko: wyniki obu torów, pozycje w rankingach, następny krok.

### Kierunki warte sprawdzenia (orientacyjnie)

- Czyszczenie: braki w kursach/składach/XML, spójność kursów między bukmacherami, marża bukmachera
  (normalizacja implikowanych prawdopodobieństw do sumy 1), duplikaty pomiarów FIFA.
- Cechy drużyn: forma (punkty, gole, różnica bramek z N ostatnich meczów, osobno dom/wyjazd),
  ranking Elo / pi-rating, bilans bezpośrednich spotkań, pozycja w tabeli przed kolejką, dni odpoczynku.
- Cechy składów: średnie/max atrybutów FIFA wyjściowej jedenastki (ostatni pomiar przed meczem),
  w podziale na formacje (Y pozycji: obrona/pomoc/atak), wiek, doświadczenie.
- Cechy z XML (`shoton`, `possession`, …) — **tylko jako średnie z poprzednich meczów**.
- Kontekst: liga, etap sezonu (`stage`), różnice cech gospodarz − gość.
- Modele `open`: gradient boosting (LightGBM / XGBoost / CatBoost), stacking, blend z modelami klasycznymi.
- Modele `classic`: LogReg i SVM (wymagają skalowania), RF / Bagging / AdaBoost na drzewach,
  MLP, a na końcu soft voting najlepszych z nich. KNN i SVC z jądrem RBF są wolne dla ~20 tys.
  wierszy — zacznij od podpróbki lub `LinearSVC`.
- **Specyfika BA:** remisy są rzadkie i trudne — kluczowe jest dobranie wag klas / progów decyzyjnych
  (np. mnożniki prawdopodobieństw klas dobierane na ostatnim sezonie treningowym).
- Tuning hiperparametrów (np. Optuna) — wyłącznie na danych treningowych/CV.
- Ensembling / stacking modeli z różnych branchy (to naturalny etap końcowy konkursu!).

---

## 10. Czego NIE robić

- ❌ Push do `main`, force-push, modyfikacja cudzych branchy.
- ❌ Modyfikacja `common/` lub `scripts/leaderboard.py` na własnym branchu, wczytywanie danych z innego źródła.
- ❌ W torze `classic`: jakikolwiek model spoza listy (sekcja 3.1), także ukryty w przetwarzaniu lub jako źródło cech;
  pomijanie `assert_classic`.
- ❌ Zaniedbanie jednego z torów — oba muszą mieć aktualny wynik w `SCORE.json`.
- ❌ Używanie sezonu 2015/2016 (holdout) do jakichkolwiek decyzji; usuwanie meczów z sezonów walidacyjnych/testowych.
- ❌ Używanie informacji z bieżącego lub przyszłych meczów jako cech (sekcja 6).
- ❌ Wpisywanie do `SCORE.json` wyniku, którego `main.py` nie odtwarza.
- ❌ Zmiana protokołu ewaluacji (folds, seed, metryka) bez uzgodnienia z zespołem — psuje porównywalność.
- ❌ Commitowanie dużych plików danych/modeli i sekretów (klucze API, tokeny).
