# ASI_ML
Opis problemu, założenia projektu, cele i osiągnięcia
**Opis problemu, założenia projektu, cele i osiągnięcia**
Projekt dotyczy analizy zużycia energii elektrycznej w czasie rzeczywistym, mającej na celu lepszą optymalizację zarządzania zasobami energetycznymi. Główne założenia obejmują:

- Przetwarzanie danych historycznych dotyczących zużycia energii.
- Tworzenie modeli predykcyjnych prognozujących zużycie w przyszłości.
- Implementację aplikacji Streamlit, która umożliwia interakcję użytkownika z modelami.
- Monitorowanie procesu modelowania i wyników za pomocą platformy wandb.

W ramach projektu udało się:
- Opracować spójny potok przetwarzania danych i trenowania modeli w oparciu o framework Kedro.
- Zaimplementować skuteczne modele predykcyjne.
- Zintegrować aplikację Streamlit z potokiem danych.
- Skonfigurować monitoring postępów uczenia modeli w wandb.

**Instrukcja instalacji i uruchomienia**

1. **Klonowanie repozytorium:**
   ```bash
   git clone https://github.com/SolarDefender/ASI_ML.git
   cd ASI_ML-main
   ```

2. **Instalacja zależności:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Uruchomienie aplikacji Streamlit:**
   ```bash
   streamlit run src/ASI_ML/pipelines/data_science/streamlit_run.py
   ```

4. **Uruchamianie potoku Kedro:**
   ```bash
   kedro run
   ```

**Struktura potoku**

Potok oparty na Kedro składa się z dwóch głównych komponentów:

1. **Data Processing:**
   - Wczytywanie i wstępne przetwarzanie danych surowych.
   - Normalizacja i transformacja danych wejściowych zgodnie z ustalonymi parametrami.

2. **Data Science:**
   - Podział danych na zbiory treningowy i testowy.
   - Trenowanie modeli (m.in. regresja, drzewa decyzyjne).
   - Ewaluacja wyników modeli.


**Opis aplikacji Streamlit i potoku Kedro**
Aplikacja Streamlit umożliwia:
- Wizualizację danych wejściowych oraz wyników modeli w czasie rzeczywistym.
- Interaktywne wybieranie parametrów modeli i ich ewaluację.
- Intuicyjne środowisko dla użytkownika, który nie musi znać szczegółów technicznych potoku.

Potok Kedro zapewnia modularność procesu przetwarzania danych oraz trenowania modeli, umożliwia to łatwą rozbudowę projektu o kolejne funkcjonalności.

**Opis monitorowania z wandb**
Platforma wandb umożliwia:
- Wizualizację kluczowych metryk, takich jak błędów predykcji czy strat treningowych.
- Zachowanie historii eksperymentów, co ułatwia analizę wyników i optymalizację procesu modelowania.

Aby uruchomić monitorowanie wandb, należy skonfigurować swoje konto i dodać klucz API w zmiennej środowiskowej:
```bash
export WANDB_API_KEY=<twój_klucz_api>
```

