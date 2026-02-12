import asyncio
import os
from datetime import datetime

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from playwright.async_api import TimeoutError as PlaywrightTimeout
from playwright.async_api import async_playwright

load_dotenv()


# --- KONFIGURACJA ---
# Wstaw tutaj swój klucz API Google Gemini
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Konfiguracja modelu VLM (Gemini 1.5 Flash jest szybki i tani do takich zadań)
model = genai.GenerativeModel("gemini-2.5-flash")

# Ścieżka do pliku CSV z usługami
CSV_FILE = "services.csv"

# Limit równoczesnych kart przeglądarki (żeby nie zabić komputera)
CONCURRENCY_LIMIT = 5


def load_urls_from_csv(csv_path):
    """
    Wczytuje URLe z pliku CSV.
    Oczekiwany format: kolumna 'Link' z URLami.
    """
    try:
        df = pd.read_csv(csv_path, delimiter=";")
        if "Link" not in df.columns:
            print(f"Błąd: Plik {csv_path} nie zawiera kolumny 'Link'")
            return []

        urls = df["Link"].dropna().tolist()
        print(f"Wczytano {len(urls)} URLi z pliku {csv_path}")
        return urls
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {csv_path}")
        return []
    except Exception as e:
        print(f"Błąd podczas wczytywania CSV: {e}")
        return []


async def analyze_screenshot_with_vlm(screenshot_bytes):
    """
    Wysyła zrzut ekranu do modelu VLM z pytaniem o status strony.
    """
    prompt = """
    Przeanalizuj ten zrzut ekranu strony internetowej. 
    
    KONTEKST: Szukamy serwisów informacyjnych do monitorowania aktualności i informacji o rynku kawy zielonej (green coffee).
    
    OCEN STRONĘ POD KĄTEM:
    - Czy to serwis INFORMACYJNY (newsy, artykuły, analizy rynkowe, raporty) czy SKLEP (sprzedaż produktów)?
    - Czy zawiera aktualne informacje o rynku kawy zielonej, cenach, trendach, uprawach?
    - Czy jest to profesjonalny serwis branżowy przydatny do codziennego monitorowania?
    
    WYKLUCZ strony typu:
    - Sklepy internetowe sprzedające kawę (B2C)
    - Strony z suplementami diety
    - Domeny parkingowe lub na sprzedaż
    
    PREFERUJ strony typu:
    - Portale branżowe z newsami o kawie
    - Importerzy publikujący raporty rynkowe
    - Platformy handlowe B2B z informacjami o cenach
    - Serwisy edukacyjne o kawie zielonej
    
    Odpowiedz w formacie JSON (bez markdown), zawierającym klucze:
    1. "status": "active", "parked", "error", lub "blocked"
    2. "site_type": Jeden z: "news_portal" (portal z newsami/artykułami), "b2b_platform" (platforma B2B/importerzy), "educational" (serwis edukacyjny), "online_shop" (sklep internetowy), "supplement_shop" (sklep ze suplementami), "corporate" (strona firmowa bez treści), "other" (inny)
    3. "is_informational": true/false (czy to serwis informacyjny, a nie sklep?)
    4. "has_market_news": true/false (czy zawiera aktualności/newsy o rynku kawy?)
    5. "is_green_coffee_focused": true/false (czy koncentruje się na kawie zielonej/surowych ziarnach?)
    6. "monitoring_value": Ocena 0-10, jak przydatna jest strona do codziennego monitorowania rynku kawy zielonej
    7. "main_content": Krótki opis głównej treści strony (1-2 zdania)
    8. "keywords_found": Lista max 5 najważniejszych słów kluczowych związanych z kawą
    9. "update_frequency": "daily" (codziennie), "weekly" (tygodniowo), "static" (rzadko aktualizowana), "unknown" (nieznana)
    10. "recommendation": "excellent" (doskonała do monitorowania), "good" (dobra), "moderate" (średnia), "poor" (słaba - to sklep/nieaktualny content)
    """

    try:
        response = await model.generate_content_async(
            [{"mime_type": "image/jpeg", "data": screenshot_bytes}, prompt]
        )
        return response.text
    except Exception as e:
        return f"VLM Error: {str(e)}"


async def check_site(context, url, semaphore):
    """
    Główna funkcja sprawdzająca pojedynczy URL.
    """
    async with semaphore:  # Pilnuje limitu równoczesnych połączeń
        # Dodanie protokołu jeśli brakuje
        target_url = url if url.startswith("http") else f"http://{url}"

        result = {
            "url": url,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "dns_exists": False,
            "http_status": None,
            "vlm_analysis": None,
            "screenshot_path": None,
            "error": None,
        }

        page = await context.new_page()
        try:
            print(f"Sprawdzam: {url}...")
            # Timeout 10 sekund na załadowanie
            response = await page.goto(
                target_url, wait_until="domcontentloaded", timeout=10000
            )

            result["dns_exists"] = True
            result["http_status"] = response.status

            # Jeśli status jest OK, robimy zrzut i analizę
            if response.status < 400:
                # Screenshot do pamięci (dla AI)
                screenshot_bytes = await page.screenshot(type="jpeg", quality=60)

                # Screenshot na dysk (dla Ciebie)
                filename = f"screens/{url.replace('http://', '').replace('https://', '').replace('/', '_')}.jpg"
                os.makedirs("screens", exist_ok=True)
                with open(filename, "wb") as f:
                    f.write(screenshot_bytes)
                result["screenshot_path"] = filename

                # Analiza VLM
                print(f" Analiza AI dla: {url}...")
                ai_response = await analyze_screenshot_with_vlm(screenshot_bytes)
                result["vlm_analysis"] = ai_response

        except PlaywrightTimeout:
            result["error"] = "Timeout"
        except Exception as e:
            error_msg = str(e)
            if "ERR_NAME_NOT_RESOLVED" in error_msg:
                result["error"] = "DNS Error (Nie istnieje)"
            else:
                result["error"] = f"Error: {error_msg[:50]}..."
        finally:
            await page.close()
            return result


def print_coffee_summary(results):
    """
    Wyświetla podsumowanie analizy stron pod kątem serwisów informacyjnych o kawie zielonej.
    """
    import json

    print("\n" + "=" * 80)
    print("📊 ANALIZA SERWISÓW - MONITORING RYNKU KAWY ZIELONEJ")
    print("=" * 80 + "\n")

    excellent_sites = []
    good_sites = []
    informational_sites = []
    shops = []

    # Ikony dla typów stron
    type_icons = {
        "news_portal": "📰",
        "b2b_platform": "🏢",
        "educational": "📚",
        "online_shop": "🛒",
        "supplement_shop": "💊",
        "corporate": "🏛️",
        "other": "❓"
    }

    # Ikony dla rekomendacji
    rec_icons = {
        "excellent": "⭐⭐⭐",
        "good": "⭐⭐",
        "moderate": "⭐",
        "poor": "❌"
    }

    for result in results:
        if result.get("vlm_analysis"):
            try:
                # Próba parsowania JSON
                analysis = result["vlm_analysis"]
                # Usuwanie markdown formatowania jeśli istnieje
                if "```json" in analysis:
                    analysis = analysis.split("```json")[1].split("```")[0].strip()
                elif "```" in analysis:
                    analysis = analysis.split("```")[1].split("```")[0].strip()

                data = json.loads(analysis)

                url = result["url"]
                site_type = data.get("site_type", "other")
                is_informational = data.get("is_informational", False)
                has_news = data.get("has_market_news", False)
                is_green_coffee = data.get("is_green_coffee_focused", False)
                monitoring_value = data.get("monitoring_value", 0)
                recommendation = data.get("recommendation", "unknown")
                content = data.get("main_content", "Brak opisu")
                update_freq = data.get("update_frequency", "unknown")

                # Wyświetlanie z kolorowym formatowaniem
                rec_icon = rec_icons.get(recommendation, "❓")
                type_icon = type_icons.get(site_type, "❓")

                print(f"{rec_icon} {url}")
                print(f"   {type_icon} Typ: {site_type.replace('_', ' ').title()}")
                print(f"   {'📄' if is_informational else '🛒'} Serwis informacyjny: {'✅ TAK' if is_informational else '❌ NIE (sklep)'}")
                print(f"   {'📊' if has_news else '⭕'} Aktualności rynkowe: {'✅ TAK' if has_news else '❌ NIE'}")
                print(f"   {'🫘' if is_green_coffee else '⭕'} Kawa zielona: {'✅ TAK' if is_green_coffee else '❌ NIE'}")
                print(f"   📈 Wartość do monitorowania: {monitoring_value}/10")
                print(f"   🔄 Częstotliwość aktualizacji: {update_freq}")
                print(f"   💬 {content}")

                if data.get("keywords_found"):
                    keywords = ", ".join(data["keywords_found"][:5])
                    print(f"   🔑 {keywords}")
                print()

                # Kategoryzacja
                if recommendation == "excellent":
                    excellent_sites.append(url)
                elif recommendation == "good":
                    good_sites.append(url)

                if is_informational:
                    informational_sites.append(url)
                else:
                    shops.append(url)

            except json.JSONDecodeError as e:
                print(f"⚠️  {result['url']} - Błąd parsowania JSON: {str(e)[:50]}")
                print()
            except Exception as e:
                print(f"⚠️  {result['url']} - Błąd: {str(e)[:50]}")
                print()

    print("=" * 80)
    print("📈 PODSUMOWANIE:")
    print(f"   ⭐⭐⭐ Doskonałe do monitorowania: {len(excellent_sites)}")
    print(f"   ⭐⭐ Dobre serwisy: {len(good_sites)}")
    print(f"   📄 Serwisy informacyjne: {len(informational_sites)}/{len(results)}")
    print(f"   🛒 Sklepy (odrzucone): {len(shops)}/{len(results)}")
    print("=" * 80 + "\n")


async def main():
    # Wczytaj URLe z pliku CSV
    urls = load_urls_from_csv(CSV_FILE)

    if not urls:
        print("Brak URLi do sprawdzenia. Kończę program.")
        return

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Ustawiamy user agent, żeby strony parkingowe nas nie blokowały
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            viewport={"width": 1280, "height": 720},
        )

        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        tasks = [check_site(context, url, semaphore) for url in urls]

        results = await asyncio.gather(*tasks)

        await browser.close()

        # Wyświetl podsumowanie analizy
        print_coffee_summary(results)

        # Zapis wyników do CSV
        df = pd.DataFrame(results)
        df.to_csv("wyniki_audytu.csv", index=False, encoding="utf-8-sig")
        print("\n✅ Zakończono! Wyniki zapisano w 'wyniki_audytu.csv'")
        print("\n📋 Podstawowe informacje:")
        print(df[["url", "http_status", "error"]])


if __name__ == "__main__":
    asyncio.run(main())
