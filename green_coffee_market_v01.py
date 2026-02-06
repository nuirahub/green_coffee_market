import json
import os
from datetime import datetime

import google.generativeai as genai

# --- KONFIGURACJA ---
API_KEY = "TWOJ_KLUCZ_API_GEMINI"  # Wstaw swój klucz API
HISTORY_FILE = "market_history.json"
REPORT_DIR = "raporty_kawa"

# Konfiguracja klienta Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")  # Używamy szybkiego modelu


class CoffeeMarketReporter:
    def __init__(self):
        self.ensure_directories()
        self.history = self.load_history()

    def ensure_directories(self):
        if not os.path.exists(REPORT_DIR):
            os.makedirs(REPORT_DIR)

    def load_history(self):
        """Ładuje historię cen z pliku JSON, aby móc liczyć trendy."""
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        return []

    def save_history(self, today_data):
        """Dopisuje dzisiejsze dane do historii."""
        # Dodajemy datę do rekordu
        today_data["date"] = datetime.now().strftime("%Y-%m-%d")
        self.history.append(today_data)
        # Trzymamy tylko ostatnie 30 dni dla porządku
        if len(self.history) > 30:
            self.history = self.history[-30:]

        with open(HISTORY_FILE, "w") as f:
            json.dump(self.history, f, indent=4)

    def get_market_data(self):
        """
        Symulacja pobierania danych rynkowych.
        W wersji produkcyjnej tutaj byłby scraper (np. BeautifulSoup) lub call do API giełdowego (np. Yahoo Finance).
        Na potrzeby demo wpisujemy dane ręcznie lub losujemy.
        """
        # Przykładowe dane "na dziś"
        return {
            "arabica_price": 307.85,  # c/lb
            "robusta_price": 3668,  # USD/t
            "usd_pln": 3.58,
            "key_news": [
                "CONAB prognozuje rekordowe zbiory w Brazylii (66.2 mln worków).",
                "Susza w Wietnamie nadal zagraża Robuście.",
                "Złoty umacnia się do dolara poniżej poziomu 3.60.",
            ],
        }

    def calculate_trends(self, current_data):
        """Oblicza zmianę w stosunku do ostatniego zapisanego raportu."""
        if not self.history:
            return "Brak danych historycznych do porównania."

        last_entry = self.history[-1]

        # Obliczenia matematyczne (Python robi to lepiej niż LLM)
        arabica_change = round(
            (
                (current_data["arabica_price"] - last_entry["arabica_price"])
                / last_entry["arabica_price"]
            )
            * 100,
            2,
        )
        robusta_change = round(
            (
                (current_data["robusta_price"] - last_entry["robusta_price"])
                / last_entry["robusta_price"]
            )
            * 100,
            2,
        )

        trend_info = f"""
        DANE HISTORYCZNE (Ostatni raport z {last_entry["date"]}):
        - Arabica wczoraj: {last_entry["arabica_price"]} (Zmiana dziś: {arabica_change}%)
        - Robusta wczoraj: {last_entry["robusta_price"]} (Zmiana dziś: {robusta_change}%)
        - USD/PLN wczoraj: {last_entry["usd_pln"]}
        """
        return trend_info

    def generate_prompt(self, data, trends):
        return f"""
        Jesteś Ekspertem Rynku Kawy Zielonej. Przygotuj profesjonalny raport dzienny w formacie MARKDOWN.
        
        DANE BIEŻĄCE:
        - Arabica (KC): {data["arabica_price"]} c/lb
        - Robusta (RC): {data["robusta_price"]} USD/t
        - Kurs USD/PLN: {data["usd_pln"]}
        
        TRENDY I HISTORIA:
        {trends}
        
        KLUCZOWE INFORMACJE (NEWSY):
        {json.dumps(data["key_news"], ensure_ascii=False)}
        
        INSTRUKCJE STRUKTURY RAPORTU (Użyj dokładnie tych nagłówków H2):
        1. ## 📊 Market Snapshot
           - Tabela z cenami i wyliczonymi zmianami procentowymi.
           - Krótki komentarz sentymentu rynku (Byka/Niedźwiedzia).
        2. ## 🌍 Kluczowe Czynniki (Drivers)
           - Opisz newsy i ich wpływ na cenę. Dodaj (fikcyjne w tym demo) linki do źródeł jako [Źródło].
        3. ## 💡 Rekomendacja Eksperta
           - Podziel na: Dziś (Spot), Tydzień, 3-Miesiące.
           - Jasna instrukcja: KUPUJ / CZEKAJ / HEDGUJ z uzasadnieniem.
           
        Ważne: Bądź konkretny, używaj języka biznesowego.
        """

    def validate_report(self, content):
        """Prosta walidacja - sprawdza czy model wygenerował kluczowe sekcje."""
        required_sections = [
            "Market Snapshot",
            "Kluczowe Czynniki",
            "Rekomendacja Eksperta",
        ]
        missing = [sec for sec in required_sections if sec not in content]

        if missing:
            return False, f"Brakuje sekcji: {', '.join(missing)}"
        return True, "OK"

    def execute(self):
        print("☕ Rozpoczynam generowanie raportu...")

        # 1. Pobierz dane
        current_data = self.get_market_data()

        # 2. Oblicz trendy
        trends = self.calculate_trends(current_data)

        # 3. Przygotuj prompt
        prompt = self.generate_prompt(current_data, trends)

        # 4. Zapytaj Gemini (Pętla walidacyjna)
        attempts = 0
        max_attempts = 2
        final_report = ""

        while attempts < max_attempts:
            print(f"🔄 Zapytanie do AI (Próba {attempts + 1})...")
            response = model.generate_content(prompt)
            report_content = response.text

            is_valid, message = self.validate_report(report_content)

            if is_valid:
                final_report = report_content
                break
            else:
                print(
                    f"⚠️ Raport niekompletny: {message}. Ponawiam z prośbą o poprawkę."
                )
                prompt += f"\n\nUWAGA: W poprzedniej odpowiedzi brakowało sekcji: {message}. Uzupełnij je proszę."
                attempts += 1

        if not final_report:
            print("❌ Nie udało się wygenerować poprawnego raportu.")
            return

        # 5. Zapisz plik Markdown
        filename = f"{REPORT_DIR}/Raport_Kawa_{datetime.now().strftime('%Y-%m-%d')}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(final_report)

        # 6. Zaktualizuj historię
        self.save_history(current_data)

        print(f"✅ Sukces! Raport zapisany w: {filename}")
        print(f"📈 Zaktualizowano bazę historyczną w: {HISTORY_FILE}")


# --- URUCHOMIENIE ---
if __name__ == "__main__":
    reporter = CoffeeMarketReporter()
    reporter.execute()
