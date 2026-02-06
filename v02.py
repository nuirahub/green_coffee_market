import json
import os
from datetime import datetime

import google.generativeai as genai

# --- KONFIGURACJA ---
API_KEY = "TWOJ_KLUCZ_API"
HISTORY_FILE = "market_history.json"
REPORT_DIR = "raporty_kawa"

# Konfiguracja modelu z dostępem do Google Search (Grounding)
genai.configure(api_key=API_KEY)

# Używamy narzędzia Google Search
tools_config = [{"google_search": {}}]
model = genai.GenerativeModel("gemini-2.0-flash", tools=tools_config)


class IntelligentCoffeeReporter:
    def __init__(self):
        self.ensure_directories()
        self.history = self.load_history()

    def ensure_directories(self):
        if not os.path.exists(REPORT_DIR):
            os.makedirs(REPORT_DIR)

    def load_history(self):
        # (Tutaj kod ładowania historii - taki sam jak poprzednio)
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        return []

    def get_hard_data(self):
        """Pobiera twarde liczby (ceny), które mamy w systemie."""
        return {"arabica_price": 307.85, "robusta_price": 3668, "usd_pln": 3.58}

    def verify_market_risks(self):
        """
        NOWOŚĆ: Autonomiczne sprawdzanie zagrożeń w Internecie.
        Gemini samo decyduje, co jest ważne.
        """
        print("🌍 Skanowanie globalnych wiadomości (Google Search)...")

        search_prompt = f"""
        Jesteś analitykiem ryzyka na rynku towarowym (kawa). 
        Data dzisiejsza: {datetime.now().strftime("%Y-%m-%d")}.
        
        ZADANIE:
        Użyj Google Search, aby znaleźć najświeższe informacje (z ostatnich 48h) dotyczące:
        1. Pogody w regionach kawowych (Minas Gerais, Central Highlands Vietnam).
        2. Logistyki morskiej (Kanał Sueski, Panamski, strajki w portach).
        3. Nagłych zmian regulacyjnych (EUDR, cła).
        
        OCENA:
        Jeśli znajdziesz coś krytycznego, co drastycznie wpłynie na cenę DZIŚ, oznacz to jako "CRITICAL".
        Jeśli to standardowy szum, oznacz jako "NORMAL".
        
        FORMAT ODPOWIEDZI (Tylko JSON):
        {{
            "status": "NORMAL" lub "CRITICAL",
            "alert_topic": "Krótki tytuł (jeśli critical) lub null",
            "description": "Opis zagrożenia (jeśli critical) lub null",
            "source_link": "Link do źródła"
        }}
        """

        # Wymuszamy odpowiedź JSON
        response = model.generate_content(
            search_prompt, generation_config={"response_mime_type": "application/json"}
        )
        try:
            return json.loads(response.text)
        except:
            return {"status": "NORMAL"}  # Fallback w razie błędu

    def generate_final_report(self, hard_data, risk_analysis):
        """Generuje raport uwzględniając wynik weryfikacji ryzyka."""

        # Bazowy prompt
        base_prompt = f"""
        Jesteś Ekspertem Rynku Kawy. Przygotuj raport dzienny w Markdown.
        Dane twarde: Arabica {hard_data["arabica_price"]}, Robusta {hard_data["robusta_price"]}, USD/PLN {hard_data["usd_pln"]}.
        """

        # Logika dynamicznej zmiany promptu
        if risk_analysis.get("status") == "CRITICAL":
            print(f"🚨 WYKRYTO KRYTYCZNE ZDARZENIE: {risk_analysis['alert_topic']}")
            dynamic_instruction = f"""
            !!! UWAGA - TRYB AWARYJNY !!!
            Wykryto krytyczne zdarzenie rynkowe: {risk_analysis["alert_topic"]} ({risk_analysis["description"]}).
            
            ZMODYFIKOWANA STRUKTURA RAPORTU:
            1. Na samym początku dodaj sekcję: # 🚨 FLASH MARKET ALERT
               Opisz zdarzenie i podaj link: {risk_analysis["source_link"]}.
               Napisz jasno: "Zalecana natychmiastowa rewizja strategii zakupowej".
            2. Dopiero potem podaj "Market Snapshot" i resztę raportu.
            3. W rekomendacji uwzględnij ten nowy czynnik jako decydujący.
            """
        else:
            print("✅ Brak krytycznych zagrożeń. Generowanie standardowego raportu.")
            dynamic_instruction = """
            Struktura standardowa:
            1. Market Snapshot (Tabela)
            2. Kluczowe Czynniki
            3. Rekomendacja
            """

        final_prompt = base_prompt + dynamic_instruction

        response = model.generate_content(final_prompt)
        return response.text

    def execute(self):
        # 1. Pobierz dane liczbowe
        hard_data = self.get_hard_data()

        # 2. Agent Weryfikacji (Skanowanie Internetu)
        risk_analysis = self.verify_market_risks()

        # 3. Generowanie Raportu (Z dynamicznym promptem)
        report_content = self.generate_final_report(hard_data, risk_analysis)

        # 4. Zapis
        filename = (
            f"{REPORT_DIR}/Raport_Inteligentny_{datetime.now().strftime('%Y-%m-%d')}.md"
        )
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"Gotowe: {filename}")


if __name__ == "__main__":
    bot = IntelligentCoffeeReporter()
    bot.execute()
