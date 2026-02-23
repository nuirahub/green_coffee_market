from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

from crewai import Agent, Crew, Process, Task

# 1. Silnik i Narzędzia
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
google_search = SerperDevTool()

# Wejdź na serper.dev.
# Zarejestruj się (dają 2500 darmowych zapytań na start, co wystarczy na tygodnie testów).
# Wklej klucz do kodu i gotowe.


# 2. Agenci (Zoptymalizowani pod Google Search)
researcher = Agent(
    role="Ekspert Google Search",
    goal="Znajdź najbardziej precyzyjne dane o {topic} korzystając z wyszukiwarki Google.",
    backstory="Jesteś mistrzem zaawansowanych operatorów Google. Potrafisz dotrzeć do PDF-ów, raportów i ukrytych stron informacyjnych.",
    tools=[google_search],
    llm=llm,
    verbose=True,
)

analyst = Agent(
    role="Analityk Strategiczny",
    goal="Przeanalizuj wyniki z Google i wyciągnij 3 kluczowe wnioski dla {topic}.",
    backstory="Potrafisz odróżnić płatne reklamy i marketing od twardych faktów biznesowych.",
    llm=llm,
    verbose=True,
)

lead_finder = Agent(
    role="Łowca Kontaktów",
    goal="Znajdź adresy email do podmiotów powiązanych z {topic}.",
    backstory="Twoim zadaniem jest przeszukanie stron znalezionych przez Badacza w celu znalezienia kontaktów.",
    tools=[google_search],
    llm=llm,
    verbose=True,
)

communicator = Agent(
    role="Specjalista Komunikacji",
    goal="Przygotuj profesjonalną wiadomość na podstawie danych z Google o {topic}.",
    backstory="Piszesz maile, które wyglądają na efekt wielogodzinnego researchu człowieka.",
    llm=llm,
    verbose=True,
)

# 3. Definicja Zadań (Taski pozostają podobne, ale są bardziej precyzyjne)
task_research = Task(
    description="Użyj Google Search, aby znaleźć najnowsze informacje o {topic}. Wytyczne: {guidelines}.",
    expected_output="Raport z listą źródeł, datami i faktami.",
    agent=researcher,
)

task_analysis = Task(
    description="Oceń wiarygodność źródeł z Google i wybierz 3 najważniejsze wątki dla {topic}.",
    expected_output="Analiza 3 kluczowych odkryć.",
    agent=analyst,
)

task_leads = Task(
    description="Znajdź dane kontaktowe (mail/LinkedIn) dla wybranych przez analityka wątków o {topic}.",
    expected_output="Lista kontaktowa w formacie tabeli.",
    agent=lead_finder,
)

task_email = Task(
    description="Napisz finalną wiadomość dotyczącą {topic}, bazując na całym zebranym procesie.",
    expected_output="Gotowy szkic profesjonalnej wiadomości.",
    agent=communicator,
)

# 4. Uruchomienie z Managerem
crew = Crew(
    agents=[researcher, analyst, lead_finder, communicator],
    tasks=[task_research, task_analysis, task_leads, task_email],
    process=Process.manager,
    manager_llm=llm,
    verbose=True,
)

# Przykład: Lokalne Budowy
result = crew.kickoff(
    inputs={
        "topic": "Nowe inwestycje logistyczne w okolicach Poznania",
        "guidelines": "Szukaj magazynów w budowie i nowych pozwoleń na budowę z 2026 roku.",
    }
)

print(result)
