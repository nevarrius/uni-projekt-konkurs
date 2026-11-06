## 2026-11-06 12:49 — [open] progi dostrojone na fragmencie treningowym → cv=0.4454 ± 0.0126 (spadek -0.0090)
- Co: dostrajanie mnożników prawdopodobieństw klas na końcowych 20% danych treningowych i ponowny fit na całym foldzie.
- Dlaczego: sprawdzenie, czy progi poprawią balanced accuracy również dla HistGradientBoosting.
- Wynik: pogorszenie CV względem wybranego wariantu z cechami kursowymi; zmianę odrzucono.

## 2026-11-06 12:48 — [classic] progi dostrojone na fragmencie treningowym → cv=0.4596 ± 0.0102 (poprawa +0.0009)
- Co: siatka mnożników prawdopodobieństw klas wybrana na chronologicznym końcowym fragmencie danych treningowych; model końcowy jest ponownie trenowany na całym foldzie.
- Dlaczego: zrównoważona dokładność wymaga uwzględnienia trudniejszej klasy remisów.
- Wynik: poprawa CV względem baseline 0.4587 ± 0.0128. Dobór nie korzysta z sezonu holdout.

## 2026-11-06 12:46 — [classic] normalizacja marży i konsensus kursów → cv=0.4548 ± 0.0113 (spadek -0.0039)
- Co: dodanie prawdopodobieństw znormalizowanych o marżę, konsensusu, rozbieżności i liczby dostępnych bukmacherów.
- Dlaczego: sprawdzenie, czy reprezentacja kursów wzbogaci regresję logistyczną.
- Wynik: CV było gorsze od baseline; `classic` wrócił do surowych cech kursowych.

## 2026-11-06 12:45 — [open] normalizacja marży i konsensus kursów → cv=0.4544 ± 0.0073 (poprawa +0.0009)
- Co: zachowanie surowych implikowanych prawdopodobieństw oraz dodanie kursów znormalizowanych do sumy 1 dla bukmachera, marży, konsensusu, rozbieżności i liczby dostępnych bukmacherów.
- Dlaczego: sprawdzenie, czy usunięcie marży i sygnał zgodności bukmacherów poprawią predykcję.
- Wynik: niewielka poprawa CV względem baseline 0.4535 ± 0.0101; zmiana pozostaje tylko w torze `open`.

## 2026-11-06 12:42 — [oba] odtworzenie baseline
- Co: uruchomienie niezmienionych rozwiązań bazowych przez wspólny `evaluate(...)`.
- Wynik: `open` 0.4535 ± 0.0101; `classic` 0.4587 ± 0.0128.
