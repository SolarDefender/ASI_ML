# ASI_ML
**Opis problemu, założenia projektu, cele i osiągnięcia**
Projekt został stworzony w celu analizy danych pogodowych, ich przekształcania oraz opracowania modelu, który umożliwi przewidywanie zużycia energii elektrycznej na podstawie nowych danych. Program może pomóc w optymalizacji zużycia energii elektrycznej w domach i firmach, umożliwiając lepsze planowanie wykorzystania urządzeń na podstawie prognoz zużycia. Dzięki temu użytkownicy mogą obniżyć rachunki za prąd i zmniejszyć negatywny wpływ na środowisko.

Główne założenia obejmują:
- Przetwarzanie danych pogodowych, które mogą mieć wpływ na zużycie energii elektrycznej.
- Tworzenie modeli predykcyjnych prognozujących zużycie w przyszłości.
- Implementację aplikacji Streamlit, która umożliwia interakcję użytkownika z modelami.
- Monitorowanie procesu modelowania i wyników za pomocą platformy wandb.

W ramach projektu udało się:
- Opracować spójny potok przetwarzania danych i trenowania modeli w oparciu o framework Kedro.
- Zaimplementować skuteczne modele predykcyjne przy użyciu AutoGluon.
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

3. **Uruchamianie potoku Kedro:**
   ```bash
   kedro run
   ```

**Struktura potoku**

Potok oparty na Kedro składa się z trzech głównych komponentów:

1. **Data Processing:**
   - Wczytywanie i wstępne przetwarzanie danych surowych przy użyciu baz danych MySQL.
   - Normalizacja i transformacja danych wejściowych zgodnie z ustalonymi parametrami.

2. **Data Science:**
   - Podział danych na zbiory treningowy i testowy.
   - Trenowanie modeli (m.in. regresja, drzewa decyzyjne).
   - Ewaluacja wyników modeli.

3. **Application Deployment:**
   - Uruchamianie API umożliwiającego wykonywanie predykcji na podstawie natrenowanego modelu i aktualizowanie zbioru danych na podstawie wyników.
   - Integracja aplikacji Streamlit z API, umożliwiająca interaktywne wizualizacje i użytkowanie przez końcowych użytkowników.


**Opis aplikacji Streamlit i potoku Kedro**
Aplikacja Streamlit umożliwia:
- Wizualizację danych wejściowych oraz wyników modeli w czasie rzeczywistym.
- Interaktywne wybieranie parametrów modeli i ich ewaluację.
- Intuicyjne środowisko dla użytkownika, który nie musi znać szczegółów technicznych potoku.

Potok Kedro zapewnia modularność procesu przetwarzania danych oraz trenowania modeli, umożliwiając łatwą rozbudowę projektu o kolejne funkcjonalności. Struktura projektu Kedro opiera się na koncepcji katalogów, które definiują źródła i przechowywanie danych, takich jak pliki CSV, bazy danych, czy inne formaty danych. Dzięki temu dane mogą być zarządzane w spójny i powtarzalny sposób w całym procesie.

**Opis monitorowania z wandb**
Platforma wandb umożliwia:
- Wizualizację kluczowych metryk, takich jak błędów predykcji czy strat treningowych.
- Zachowanie historii eksperymentów, co ułatwia analizę wyników i optymalizację procesu modelowania.

Aby uruchomić monitorowanie wandb, należy skonfigurować swoje konto i dodać klucz API w zmiennej środowiskowej:
```bash
export WANDB_API_KEY=<twój_klucz_api>
```

