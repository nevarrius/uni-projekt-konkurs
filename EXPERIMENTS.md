# Dziennik eksperymentów — agent/oskar

Uwaga metodologiczna: różnice BA < ~0.005 to szum (fold ≈ 3 tys. meczów → σ(BA) ≈ 0.009).
Oprócz oficjalnego CV (3 foldy) patrzę też na **CV5** — walk-forward z 5 foldami (walidacja 2010/11–2014/15,
tylko sezony treningowe, bez holdoutu) — i wybieram konfiguracje dobre w obu.

## 2026-11-06 16:00 — [oba] soft voting LogReg+RF+MLP (classic) i + CatBoost + HistGB (open)  →  classic cv=0.4703 ± 0.0064, open cv=0.4709 ± 0.0062
- classic: VotingClassifier(soft) LogReg(C=.05) + RF(500, leaf=30, mf=.3) + MLP(16, alpha=10). CV5=0.4649. Holdout 0.4645.
- open: (3·P_voting + P_CatBoost + P_HGB)/5, decyzja p/prior. CV5=0.4656. Holdout 0.4622.
- RF sam: CV3 0.4670, CV5 0.4652 (najlepszy pojedynczy model w CV5!). Strojenie RF (leaf 15/60, mf .15, wszystkie kolumny) — bez poprawy.
- 2×RF w votingu: CV3 0.4709 / CV5 0.4649 — w szumie, zostają równe wagi.
- Czas pełnego `main.py`: ~2:45 min.

## 2026-11-06 15:30 — [oba] porażki / neutralne (nie traćcie czasu)
- **Strojenie wag klas na wewnętrznej walidacji** (ostatni sezon treningowy, siatka w_D, w_A): 0.4626 vs 0.4686 dla czystego p/prior — przeuczenie progów.
- **Wskaźniki braków** (SimpleImputer add_indicator) w LogReg: 0.4585 (−0.010).
- **Osobny model dla meczów bez kursów** (ligi PL 15722 i CH 24558 nie mają ŻADNYCH kursów, 13% meczów): gorzej.
  Trening modelu z kursami tylko na meczach z kursami: −0.009 na podzbiorze z kursami — mecze bez kursów w treningu pomagają.
- **Time-decay wag próbek** (0.85/0.7/0.5 na sezon): monotonicznie gorzej.
- **One-hot ligi** w LogReg: 0.4629 (gorzej). **Ordinal logit** (Frank & Hall, 2×LogReg): 0.4665 (bez zysku).
- **Cechy pochodne pod remisy** (|różnice sił|, P(remis) z Poissona z formy EWM, oczekiwana suma goli): LogReg 0.4650 vs 0.4686 — bez zysku; zostawione tylko dla boostingu w open.
- MLP(32, alpha=1) słaby (0.4537); MLP(16, alpha=10) OK (0.4663).
- Kompaktowy zestaw ~40 cech-różnic: CV3 0.4694, ale CV5 gorsze (0.4592) — niejednoznaczne.
- LightGBM: na macOS wymaga `brew install libomp` → pominięty dla odtwarzalności; CatBoost + HGB zamiast.

## 2026-11-06 15:00 — [classic] LogReg(C=0.05) na pełnym zestawie cech + decyzja p/prior  →  cv=0.4686 ± 0.0070
- Cechy (solutions/features.py, 167 kolumn, ~3 s):
  - kursy 10 bukmacherów → znormalizowane prawdopodobieństwa, **metoda Shina** usuwania marży (Štrumbelj 2014),
    najlepsze kursy rynku, std między bukmacherami (niepewność), marża, log(pH/pA), entropia;
  - **Elo** z mnożnikiem marginesu bramek i regresją 20% do średniej między sezonami; **pi-ratings**
    (Constantinou & Fenton 2013) — osobne ratingi dom/wyjazd; mecze z tego samego dnia nie widzą się nawzajem;
  - forma (rolling 5/10, EWM) punktów, goli, GD, remisów; forma dom/wyjazd; dni odpoczynku; punkty/ppg w sezonie,
    percentyl w tabeli; H2H (ostatnie 6); strzały celne i posiadanie z XML **tylko z poprzednich meczów**;
  - atrybuty FIFA wyjściowej 11 (merge_asof, data pomiaru < data meczu): średnia/max/min overall, potential, wiek,
    per formacja (GK/obrona/pomoc/atak po Y) + atak vs obrona rywala; Team_Attributes (ostatni pomiar < mecz);
  - kontekst: liga, kolejka, miesiąc, historyczny % remisów i zwycięstw gospodarzy w lidze (poprzednie sezony).
- **Kluczowe: reguła decyzyjna argmax p_k/prior_k** (optymalna dla BA przy skalibrowanych prawdop.): +0.02–0.025
  względem argmax we wszystkich modelach (np. LogReg odds-only 0.4331 → 0.4603).
- Ablacja (usunięcie grupy z LogReg): kursy −0.006, składy −0.006, H2H −0.004, forma −0.003, team −0.003, ratingi ~0.
- Porównanie modeli (wszystkie cechy, p/prior): LogReg 0.4686, CatBoost 0.4644, HGB 0.4644.
- Inspiracje: Hubáček, Šourek, Železný (2019) „Learning to predict soccer results from relational data with gradient
  boosted trees” (pi-ratings + GBT, zwycięzcy 2017 Soccer Prediction Challenge); Constantinou & Fenton (2013) pi-ratings;
  Štrumbelj (2014) o usuwaniu marży z kursów.
