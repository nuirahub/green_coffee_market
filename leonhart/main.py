import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from openai import OpenAI

# Ładowanie zmiennych środowiskowych i konfiguracji
load_dotenv()

# Inicjalizacja klienta OpenAI (klucz pobierany automatycznie z .env)
client = OpenAI()

# Określenie ścieżki bazowej (katalog, w którym znajduje się ten skrypt)
SCRIPT_DIR = Path(__file__).parent


def load_prompts(filename="prompts.yaml"):
    # Ścieżka względem lokalizacji skryptu, a nie CWD
    filepath = SCRIPT_DIR / filename
    with open(filepath, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_week_dates():
    today = datetime.date.today()
    days_ahead = 0 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    start_date = today + datetime.timedelta(days=days_ahead)
    end_date = start_date + datetime.timedelta(days=6)

    # Generujemy listę dat dla każdego dnia do promptu
    dates_map = {}
    for i in range(7):
        d = start_date + datetime.timedelta(days=i)
        dates_map[i] = d.strftime("%d.%m")

    return start_date.strftime("%d.%m.%Y"), end_date.strftime("%d.%m.%Y"), dates_map


def search_web_for_news(start_date, end_date):
    """
    Wyszukiwanie 'High-Level' dla grupy docelowej Premium Male 35-45.
    Ignorujemy lokalne wydarzenia. Szukamy globalnych/krajowych trendów.
    """
    # Ustalanie miesiąca i roku dla precyzji
    start_dt = datetime.datetime.strptime(start_date, "%d.%m.%Y")
    current_month_name = start_dt.strftime(
        "%B"
    )  # Np. February (lub polska nazwa jeśli masz locale)
    year = start_dt.year

    print(f"🌍 Skanuję trendy dla Mężczyzny Premium ({start_date} - {end_date})...")

    results_text = f"--- RAPORT TRENDÓW (Daty: {start_date}-{end_date}) ---\n"

    # ZAPYTANIA PRECYZYJNE (To jest klucz do sukcesu)
    # Zamiast "wydarzenia kulturalne", pytamy o "hity kinowe" i "trendy biznesowe"
    queries = {
        "BIZNES_TECH": f"najważniejsze trendy technologiczne AI biznes {year} luty",
        "KINO_SERIALE": f"premiery kinowe hity polska {year} luty",  # Szukamy hitów, nie niszowego kina
        "SPORT_PREMIUM": f"terminarz wydarzeń sportowych tenis f1 piłka nożna luty {year}",
        "LIFESTYLE": f"tematy rozmów biznesowych luty {year} forbes business insider",
    }

    with DDGS() as ddgs:
        for category, q in queries.items():
            print(f"   -> Pobieram wsad dla kategorii: {category}...")
            results_text += f"\n### KATEGORIA: {category} ###\n"

            try:
                # Używamy ddgs.text() ale z bardzo precyzyjnym zapytaniem,
                # bo ddgs.news() czasem daje zbyt dużo drobnicy.
                results = list(
                    ddgs.text(q, region="pl-pl", timelimit="m", max_results=4)
                )

                for r in results:
                    title = r["title"]
                    snippet = r["body"]
                    # Prosty filtr anty-spamowy w Pythonie:
                    # Jeśli tytuł zawiera nazwę małego miasta lub słowa "dom kultury", "warsztaty dla dzieci" -> pomiń
                    blacklist = [
                        "dom kultury",
                        "warsztaty",
                        "biblioteka",
                        "gminny",
                        "powiatowy",
                        "dla dzieci",
                    ]
                    if any(bad_word in title.lower() for bad_word in blacklist):
                        continue

                    results_text += f"- {title}: {snippet}\n"
            except Exception as e:
                print(f"   [!] Błąd: {e}")

    return results_text


def call_ai(system_msg, user_msg, model="gpt-4o"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Błąd: {e}"


def main():
    prompts = load_prompts()
    start_date, end_date, dates_map = get_week_dates()

    print(f"\n☕ GENERATOR POSTÓW PREMIUM + NEWSJACKING ({start_date} - {end_date})\n")

    # --- KROK 0: RESEARCH (Python szuka w sieci) ---
    news_data = search_web_for_news(start_date, end_date)

    # --- KROK 1: PROPOZYCJE TEMATÓW (AI analizuje znalezione newsy) ---
    print("\nAnalizuję znalezione newsy i generuję propozycje...")

    prompt_research = prompts["step1_research"].format(
        current_date=datetime.date.today(),
        search_results=news_data,
        start_date=start_date,
        end_date=end_date,
        date_mon=dates_map[0],
        date_tue=dates_map[1],
        # Można dodać resztę dni, tu skrótowo
    )

    suggested_topics = call_ai(
        prompts["system_role"], prompt_research, prompts["config"]["model"]
    )

    print("\n--- ZAPROPONOWANE TEMATY NA BAZIE NEWSÓW ---")
    print(suggested_topics)
    print("--------------------------------------------")

    # --- KROK 2: INTERWENCJA I GENEROWANIE ---
    print("\nCzy akceptujesz tematy? (Wpisz 'tak' lub dopisz własne uwagi):")
    user_input = input("> ")

    final_topics = (
        suggested_topics
        if user_input.lower() in ["tak", "t", ""]
        else f"{suggested_topics}\nUWAGI UŻYTKOWNIKA: {user_input}"
    )

    print("\nGeneruję finalny plan...")
    prompt_content = prompts["step2_create_content"].format(topics_list=final_topics)
    final_plan = call_ai(
        prompts["system_role"], prompt_content, prompts["config"]["model"]
    )

    filename = f"PLAN_NEWS_{start_date}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_plan)

    print(f"\n✅ Gotowe! Zapisano w {filename}")


if __name__ == "__main__":
    main()
