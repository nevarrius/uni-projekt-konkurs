# Dziennik eksperymentów — agent/konrad

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
