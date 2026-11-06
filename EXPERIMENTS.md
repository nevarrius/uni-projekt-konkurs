# Dziennik eksperymentów — agent/konrad

## 2026-11-06 14:40 — [oba] zwarty zestaw cech: kursy + FIFA składów + kontekst  →  classic cv=0.4726 ± 0.0062, open cv=0.4726 ± 0.0062  (poprawa +0.005 / +0.007)
- Co: pełne przeszukanie podzbiorów grup cech (odds zawsze + {elo, form, player, team, ctx, new}) dla LR C∈{0.003, 0.01}.
  Top: odds+player+ctx 0.4722, odds+player 0.4713; dodanie elo/form/team zawsze lekko pogarsza → są redundantne z kursami.
  Uwaga: wybór spośród 128 podzbiorów = pewien optymizm selekcji, ale wzorzec „odds+player” stabilny w top-5.
- Modele na zwartym zestawie: LR 0.4726 | MLP(16) x5 seedów 0.4697 | CatBoost 0.4681 | RF d8 0.4652 | HGB 0.4630 |
  CB+LR 0.4701 | LR+MLPx5 0.4702 | CB+LR+MLPx5 0.4694 | CB(d3)+LR 0.4677. Sama LR najlepsza → open = LR (jak classic).
- PORAŻKI:
  - Osobny model (bez kursów) dla lig bez kursów (Polska 15722, Szwajcaria 24558 — 0% kursów, ~13% meczów; BA tam ~0.41 vs ~0.48):
    0.464–0.466, gorzej niż jeden model z imputacją + flagą.
  - Dynamiczny model Poissona (siły ataku/obrony dom/wyjazd → P(H/D/A)), ligowy wskaźnik remisów, cechy |Δ| (bliskość sił): LR 0.468→0.465, CB 0.466→0.465.
  - CatBoost (różne depth/iters) na pełnym zestawie 0.4655–0.4669.

## 2026-11-06 13:40 — [oba] nowe cechy (kursy bez marży, Elo, forma, tabela, H2H, FIFA składów/drużyn)  →  open cv=0.4652 ± 0.0074, classic cv=0.4680 ± 0.0048  (poprawa +0.012 / +0.009)
- Co: `solutions/features.py` — 120 cech: średnie prawdop. z 10 bukmacherów po usunięciu marży (+std, entropia, flaga braku),
  Elo z mnożnikiem za różnicę bramek i regresją 20% między sezonami, forma (pkt/gole/remisy l5/l10/ewm, dom/wyjazd osobno),
  dni odpoczynku, ppg/gd w sezonie, pozycja w tabeli, H2H (ost. 6), ratingi FIFA wyjściowej 11 (mean/max/min, wg formacji z Y),
  atrybuty taktyczne drużyn (merge_asof, data < mecz), stage_frac, liga, miesiąc. Wszystko z `shift`/stanu przed meczem.
- classic: LogReg balanced, C sweep {0.0005..1}: najlepsze C=0.001–0.003 (0.468); C=0.05 → 0.463. Silna regularyzacja kluczowa.
- open: HGB balanced 0.4652 (early stopping domyślnie włączony, więc max_iter bez znaczenia).
- PORAŻKI: mnożniki prawdop. klas strojone na ostatnim sezonie treningowym → GORZEJ (LR 0.463→0.453, HGB 0.465→0.457) — za mało danych, szum.
  RF (best: max_depth=8, leaf=10) 0.4664, MLP (32,8) a=3 0.4677 — nie lepsze od LR. Soft-blendy LR+RF, LR+MLP, LR+RF+MLP ≈0.466, HGB+LR 0.458 — nie pomagają.
- Ablacja (usunięcie grupy, LR/HGB): -odds 0.454/0.457, -player 0.458/0.460, -elo 0.460/0.463, -ctx 0.461/0.461, -form 0.465/0.462, -team 0.465/0.464.
  Najważniejsze: kursy > FIFA graczy > Elo.
- LightGBM na macOS wymaga systemowego libomp — odpuszczone dla odtwarzalności.

## 2026-11-06 12:50 — [oba] baseline  →  open cv=0.4535 ± 0.0101, classic cv=0.4587 ± 0.0128
- Co: odtworzenie baseline z `main` (implikowane prawdop. z kursów 6 bukmacherów).
- Wynik: open (HGB) 0.4535, holdout 0.4509; classic (LogReg balanced) 0.4587, holdout 0.4565.
