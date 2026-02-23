import os

# --- 1. IMPORT NARZĘDZI ---
from crewai_tools import TavilySearchResults
from langchain_community.tools import Tool

# Uwaga: Dla Google Custom Search często używa się LangChain GoogleSearchAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_openai import ChatOpenAI

from crewai import Agent, Crew, Process, Task

# NarzędzieKluczowa zaletaZastosowanie w Twoim projekcieTavilyZwraca gotowy tekst/kontekstIdealne do Nowinek, gdzie liczy się treść artykułu.FirecrawlPrzeszukuje i "skrobuje" do MarkdownNajlepsze dla Budów, by wyciągnąć tabele z danymi.Google CustomOficjalne wyniki GoogleStabilne dla Giełdy i oficjalnych komunikatów.
# Które narzędzie wybrać do którego zadania?
# Tavily (TavilySearchResults):

# Kiedy: Chcesz, aby agent dostał od razu "streszczenie" treści strony. Tavily samo decyduje, co jest ważne.

# Zaleta: Bardzo rzadko "zapycha" kontekst LLM śmieciami (kodem HTML, reklamami).

# Firecrawl (FirecrawlSearchTool):

# Kiedy: Musisz "wejść głębiej". Firecrawl nie tylko podaje linki, ale potrafi zmapować całą domenę i zamienić ją na czysty tekst Markdown.

# Zaleta: Idealne, gdy Twoim źródłem są skomplikowane strony urzędowe (BIP) lub portale branżowe z dużą ilością tekstu.

# Google Custom Search:

# Kiedy: Masz już listę zaufanych stron (np. tylko money.pl, gpw.pl) i chcesz szukać wyłącznie tam.

# Zaleta: Pełna kontrola nad tym, jakie domeny przeszukuje agent (możesz ograniczyć wyszukiwanie tylko do domen .gov.pl).

# Uwaga techniczna:
# Aby Firecrawl działał poprawnie w Twoim flow, musisz go zainstalować komendą:
# pip install firecrawl-py crewai[tools]


# Konfiguracja kluczy API (ustaw w systemie lub tutaj)
os.environ["TAVILY_API_KEY"] = "tvly-..."
os.environ["FIRECRAWL_API_KEY"] = "fc-..."
os.environ["GOOGLE_CSE_ID"] = "..."  # ID Twojej wyszukiwarki
os.environ["GOOGLE_API_KEY"] = "..."

llm = ChatOpenAI(model="gpt-4o")

# --- 2. DEFINICJA NARZĘDZI (Wybierz jedno dla Badacza) ---

# OPCJA A: Tavily (AI-native search)
tavily_tool = TavilySearchResults(k=5)

# OPCJA B: Firecrawl (Search + Scrape to Markdown)
# Wymaga: pip install firecrawl-py
from crewai_tools import FirecrawlSearchTool

firecrawl_tool = FirecrawlSearchTool()

# OPCJA C: Google Custom Search (Oficjalne)
search = GoogleSearchAPIWrapper()
google_custom_tool = Tool(
    name="google_search",
    description="Wyszukuje informacje w Google.",
    func=search.run,
)

# --- 3. ARCHITEKTURA AGENTÓW ---


def create_crew(selected_tool, topic, guidelines):

    # Badacz z wybranym przez Ciebie narzędziem
    researcher = Agent(
        role="Specjalista Researchu",
        goal=f"Znajdź dane o {topic}",
        backstory="Jesteś ekspertem od pozyskiwania danych wysokiej jakości.",
        tools=[selected_tool],  # Tutaj trafia wybrane narzędzie
        llm=llm,
        verbose=True,
    )

    analyst = Agent(
        role="Analityk",
        goal="Wybierz 3 kluczowe fakty",
        backstory="Twoim zadaniem jest selekcja i weryfikacja.",
        llm=llm,
    )

    # ... reszta agentów (Lead Finder, Communicator) jak w poprzednim kodzie ...

    # Zadania
    task1 = Task(
        description=f"Research: {topic}. Wytyczne: {guidelines}",
        agent=researcher,
        expected_output="Raport",
    )
    task2 = Task(
        description="Analiza wyników", agent=analyst, expected_output="3 punkty"
    )

    return Crew(
        agents=[researcher, analyst],
        tasks=[task1, task2],
        process=Process.manager,
        manager_llm=llm,
    )


# --- 4. URUCHOMIENIE POSZCZEGÓLNYCH WARIANTÓW ---

# Przykład 1: Nowinki technologiczne za pomocą Tavily
# print("--- URUCHAMIAM TAVILY ---")
# crew_tavily = create_crew(tavily_tool, "AI News 2026", "Szukaj LLM")
# crew_tavily.kickoff()

# Przykład 2: Budowy za pomocą Firecrawl (wyciąga głęboką treść stron)
print("--- URUCHAMIAM FIRECRAWL ---")
crew_firecrawl = create_crew(
    firecrawl_tool, "Budowa magazynów Poznań", "Szukaj pozwoleń"
)
result = crew_firecrawl.kickoff()

# Przykład 3: Giełda za pomocą Google Custom Search
# print("--- URUCHAMIAM GOOGLE CUSTOM SEARCH ---")
# crew_google = create_crew(google_custom_tool, "Akcje Apple prognozy", "Szukaj raportów 10-K")
# crew_google.kickoff()
