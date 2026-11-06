# Dziennik eksperymentów — agent/nevarious

Najnowsze na górze. BA = balanced accuracy, CV = średnia z 3 foldów walk-forward, HO = holdout 2015/2016.
Szum: SE BA jednego sezonu ≈ 0.009, średniej z 3 foldów ≈ 0.005 — różnice < ~0.005 traktuję jako szum.
Do porównań pomocniczo używam log-loss na prawdopodobieństwach OOF (mniej zaszumiony niż BA).

## 2026-11-06 — [oba] kursy + destylacja kursów + końcowa LogReg  →  classic cv=0.4706, open cv=0.4689
- Cechy (`solutions/features.py`, 177 kolumn, wszystko sprzed meczu): znormalizowane kursy 10 bukmacherów
  (średnia/std/max, marża, entropia), Elo (K=20, przewaga domu 65, regres 20% między sezonami, mnożnik
  różnicy bramek), forma 3/5/10/20 meczów, forma dom/wyjazd, sezon do tej pory, dni odpoczynku, H2H,
  pozycja w tabeli przed kolejką, oceny FIFA wyjściowych 11 (merge_asof ściśle przed datą meczu,
  średnia/max/min/std, strefy GK/DEF/MID/ATT wg współrzędnej Y), wiek, wzrost, atrybuty drużyn FIFA, liga, etap.
- **Kluczowe odkrycie 1:** same kursy z regułą `argmax p/prior` dają BA ≈ 0.479 na meczach z kursami
  (vs ~0.44 przy zwykłym argmax). Żaden model na wszystkich cechach nie pobił kursów na tych meczach.
- **Kluczowe odkrycie 2:** Polska i Szwajcaria nie mają kursów w ogóle (~13% meczów), tam modele miały BA ~0.40.
- **Destylacja:** model uczony odtwarzać prawdopodobieństwa bukmachera z cech bez kursów (tylko na
  treningowych meczach z kursami). Oceniony na WSZYSTKICH meczach walidacyjnych (udając brak kursów):
  BA 0.465 vs 0.450 dla klasyfikatora uczonego na wynikach — cel jest dużo mniej zaszumiony.
- Końcowa LogReg na log(q) (q = kursy lub destylacja) + interakcje z flagą „brak kursów”.
- classic: destylacja = LogisticRegression na miękkich etykietach (wiersz ×3, wagi = prawdop. bukmachera).
- open: destylacja = średnia (3× LGBMRegressor, LogReg soft).
- Wynik: classic CV 0.4706 ± 0.007 (HO 0.4645), open CV 0.4689 ± 0.007 (HO 0.4672).

## Porażki / ślepe zaułki (nie powtarzać)
- LogReg/LightGBM/RF bezpośrednio na wszystkich 177 cechach: CV 0.456–0.461 — gorsze niż kursy + reguła p/prior.
- Strojenie wag decyzyjnych klas (poza p/prior) na treningu: wagi wychodzą ≈1, na walidacji ±0.004 — szum.
- Klasyfikator trenowany tylko na ligach bez kursów (PL+CH): gorszy niż na wszystkich ligach.
- Destylacja MLPRegressor (64) na log-prawdop.: gorsza od LogReg soft (BA bez kursów 0.459 vs 0.463).
- LightGBM na wszystkich cechach: więcej drzew = gorzej (n=200: 0.4645, n=1000: 0.4545).
