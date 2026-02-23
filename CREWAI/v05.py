import os

from langchain_community.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper

from crewai import Agent, Task

# Ustawienia kluczy
os.environ["GOOGLE_CSE_ID"] = "Twoje_ID_z_panelu_Google"
os.environ["GOOGLE_API_KEY"] = "Twoj_Klucz_API"

# Inicjalizacja silnika
search = GoogleSearchAPIWrapper()


# Definicja narzędzia z "instrukcją" filtrowania
def limited_google_search(query):
    # Dodajemy automatycznie operator 'site:' do każdego zapytania
    # Możesz to zmieniać dynamicznie zależnie od procesu
    restricted_query = f"{query} site:gov.pl OR site:gpw.pl"
    return search.run(restricted_query)


google_filter_tool = Tool(
    name="google_official_search",
    description="Wyszukuje informacje WYŁĄCZNIE w oficjalnych polskich serwisach rządowych i giełdowych.",
    func=limited_google_search,
)

# Agent Badacz wykorzystujący filtr
researcher = Agent(
    role="Ekspert ds. Dokumentacji Oficjalnej",
    goal="Znajdź oficjalne komunikaty i decyzje na temat {topic}",
    backstory="Jesteś specjalistą od białego wywiadu (OSINT) pracującym wyłącznie na źródłach o najwyższym stopniu zaufania.",
    tools=[google_filter_tool],
    verbose=True,
)

# Zadanie dla procesu "Budowy"
task_official_check = Task(
    description=(
        "Sprawdź w rejestrach urzędowych (BIP, gov.pl) status inwestycji: {topic}. "
        "Szukaj numerów decyzji, dat wydania pozwoleń i nazw inwestorów."
    ),
    expected_output="Raport urzędowy z podaniem konkretnych adresów stron .gov.pl",
    agent=researcher,
)
