import os

import dns.resolver
from crewai_tools import FileReadTool, SerperDevTool
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

from crewai import Agent, Crew, Process, Task

# 1. KONFIGURACJA
os.environ["GOOGLE_API_KEY"] = "TWOJ_KLUCZ_GEMINI"
os.environ["SERPER_API_KEY"] = "TWOJ_KLUCZ_SERPER"

# Modele: Pro dla Managera (logika), Flash dla Agentów (szybkość)
llm_manager = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)
llm_agent = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

# --- FUNKCJE (TOOLS) ---


def verify_dns_logic(email_or_domain):
    """Logika sprawdzająca istnienie serwerów pocztowych (MX) dla domeny."""
    domain = email_or_domain.split("@")[-1].strip()
    try:
        dns.resolver.resolve(domain, "MX")
        return f"WYNIK: Domena {domain} jest AKTYWNA i posiada serwery pocztowe."
    except Exception as e:
        return f"WYNIK: Domena {domain} jest NIEAKTYWNA lub błędna. Błąd: {e}"


dns_tool = Tool(
    name="Weryfikator_DNS",
    func=verify_dns_logic,
    description="Użyj do technicznego sprawdzenia czy domena email istnieje w rejestrach internetowych.",
)


def image_gen_logic(prompt):
    """Logika symulująca generowanie obrazu (można tu wpiąć API DALL-E)."""
    print(
        f"\n[SYSTEM-INFO]: Wysyłanie zapytania do generatora obrazów z opisem: {prompt}"
    )
    return f"SUKCES: Obraz wygenerowany na podstawie opisu: {prompt[:50]}..."


img_tool = Tool(
    name="Generator_Obrazow",
    func=image_gen_logic,
    description="Użyj do stworzenia wizualizacji na podstawie szczegółowego opisu (promptu).",
)

# --- AGENTORZY I ICH PROMPTY ---

analyst = Agent(
    role="Starszy Analityk Danych i Dokumentacji",
    goal="Precyzyjne wyciąganie kluczowych informacji z plików i internetu.",
    backstory="""Jesteś skrupulatnym analitykiem z 10-letnim doświadczeniem. 
    Twoim zadaniem jest dostarczanie twardych faktów. Jeśli czytasz plik, skupiasz się na 
    konkretnych wymaganiach biznesowych. Jeśli szukasz w Google, odrzucasz reklamy, 
    wybierając tylko rzetelne źródła i aktualne trendy.""",
    tools=[FileReadTool(), SerperDevTool()],
    llm=llm_agent,
    verbose=True,
)

verifier = Agent(
    role="Inżynier Bezpieczeństwa i Walidacji",
    goal="Eliminacja błędnych danych i potwierdzanie autentyczności adresów email.",
    backstory="""Jesteś technicznym 'strażnikiem'. Nie ufasz danym wygenerowanym przez AI, 
    dopóki nie sprawdzisz ich narzędziem DNS. Twoim celem jest upewnienie się, że 
    zespół nie wyśle maili w próżnię (na nieistniejące domeny).""",
    tools=[dns_tool],
    llm=llm_agent,
    verbose=True,
)

creative = Agent(
    role="Dyrektor Kreatywny i Copywriter",
    goal="Tworzenie perswazyjnej komunikacji i wysokiej jakości promptów graficznych.",
    backstory="""Jesteś mistrzem słowa i obrazu. Potrafisz przekształcić suche dane 
    techniczne w tekst, który sprzedaje. Dodatkowo, świetnie rozumiesz jak działają 
    modele generatywne (jak Midjourney czy DALL-E), dlatego Twoje opisy wizualne 
    są niezwykle szczegółowe i artystyczne.""",
    tools=[img_tool],
    llm=llm_agent,
    verbose=True,
)

# --- ZADANIA I ICH PROMPTY ---

task_1 = Task(
    description="""1. Odczytaj plik 'prompt.txt' z lokalizacji współdzielonej.
    2. Przeprowadź research w Google Search, aby uzupełnić dane z pliku o najnowsze 
    trendy z tego tygodnia w branży klienta. Wybierz 3 najciekawsze wątki.""",
    expected_output="Zestawienie danych z pliku połączone z 3 trendami z internetu.",
    agent=analyst,
)

task_2 = Task(
    description="""Sprawdź listę domen/maili dostarczonych w dokumentacji: ['microsoft.com', 'test-fake-123.pl', 'google.com'].
    Użyj narzędzia DNS, aby zweryfikować ich poprawność. Przygotuj listę 'bezpiecznych' adresów.""",
    expected_output="Lista zweryfikowanych domen z oznaczeniem ich statusu.",
    agent=verifier,
)

task_3 = Task(
    description="""Na podstawie researchu (Task 1) i zweryfikowanych kontaktów (Task 2):
    1. Napisz treść maila biznesowego (pamiętaj o profesjonalnym tonie).
    2. Przygotuj prompt dla Generatora_Obrazow, który stworzy grafikę pasującą do tego maila.
    3. Uruchom narzędzie Generator_Obrazow.""",
    expected_output="Kompletny mail, prompt graficzny oraz raport z generacji obrazu.",
    agent=creative,
)

# --- ORKIESTRATOR (MANAGEMENT) ---

#

crew = Crew(
    agents=[analyst, verifier, creative],
    tasks=[task_1, task_2, task_3],
    process=Process.hierarchical,
    manager_llm=llm_manager,  # To jest "mózg" operacji
    verbose=True,
)

# URUCHOMIENIE
if __name__ == "__main__":
    print("### START ORKIESTRACJI ZADAŃ ###")
    wynik = crew.kickoff()
    print("\n\n####################################")
    print("FINALNY RAPORT SYSTEMU:")
    print(wynik)
