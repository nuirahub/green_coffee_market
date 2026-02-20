import os

import dns.resolver
from crewai_tools import FileReadTool, SerperDevTool
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

from crewai import Agent, Crew, Process, Task

# 1. KONFIGURACJA API
os.environ["GOOGLE_API_KEY"] = "TWOJ_KLUCZ_GEMINI"
os.environ["SERPER_API_KEY"] = "TWOJ_KLUCZ_SERPER"  # Opcjonalne do Google Search

# Inicjalizacja modelu Gemini
# Manager potrzebuje mocniejszego modelu (pro), agenci mogą działać na flash
llm_manager = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)
llm_agent = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)


# 2. DEFINICJA NARZĘDZI (CUSTOM TOOLS)
def verify_dns(email):
    try:
        domain = email.split("@")[-1].strip()
        dns.resolver.resolve(domain, "MX")
        return f"Domena {domain} jest poprawna."
    except:
        return f"Domena {domain} NIE ISTNIEJE."


dns_tool = Tool(
    name="DNS_Validator",
    func=verify_dns,
    description="Sprawdza czy domena email istnieje w rzeczywistości.",
)


def image_gen_mock(prompt):
    print(f"\n[SYSTEM]: Generowanie obrazu dla promptu: {prompt}")
    return "image_generated_v1.png"


img_tool = Tool(
    name="Image_Generator",
    func=image_gen_mock,
    description="Generuje grafikę na podstawie opisu wizualnego.",
)

# Narzędzia standardowe
file_tool = FileReadTool()
search_tool = SerperDevTool()

# 3. DEFINICJA AGENTÓW (EKSPERCI)
# Nie przypisujemy im narzędzi tutaj, Manager im je przydzieli w trybie hierarchicznym
# lub przypisujemy te, które są ich "specjalizacją".

analyst = Agent(
    role="Specjalista Dokumentacji",
    goal="Pobieranie i analiza danych z plików lokalnych oraz wyszukiwarki.",
    backstory="Jesteś ekspertem od zbierania faktów. Potrafisz czytać pliki i przeszukiwać internet.",
    tools=[file_tool, search_tool],
    llm=llm_agent,
    verbose=True,
)

verifier = Agent(
    role="Audytor Techniczny",
    goal="Weryfikacja poprawności domen i odsiewanie halucynacji AI.",
    backstory="Twoim zadaniem jest upewnienie się, że każdy adres email jest prawdziwy.",
    tools=[dns_tool],
    llm=llm_agent,
    verbose=True,
)

creative = Agent(
    role="Copywriter i Artysta AI",
    goal="Tworzenie treści maili i generowanie promptów wizualnych.",
    backstory="Łączysz dane analityczne z kreatywnym pisaniem i sztuką generatywną.",
    tools=[img_tool],
    llm=llm_agent,
    verbose=True,
)

# 4. DEFINICJA ZADAŃ (FLOW)
task_1_local = Task(
    description="Pobierz dane z pliku 'dane_biznesowe.txt' (znajduje się w głównym folderze) i podsumuj kluczowe wymagania klienta.",
    expected_output="Krótka lista wymagań wyekstrahowana z pliku.",
    agent=analyst,  # Sugestia agenta, Manager może to zmienić
)

task_2_verify = Task(
    description="Sprawdź listę serwisów: ['google.com', 'firma-widmo.pl', 'microsoft.com']. Ustal, które domeny są prawdziwe.",
    expected_output="Raport techniczny poprawności domen.",
    agent=verifier,
)

task_3_creative = Task(
    description="""Na podstawie wymagań z Task 1 oraz raportu z Task 2: 
    1. Przygotuj treść maila ofertowego. 
    2. Stwórz prompt dla grafiki promującej tę ofertę.
    3. Wywołaj generator obrazu.""",
    expected_output="Gotowy mail, link do obrazu oraz uzasadnienie wyboru treści.",
    agent=creative,
)

# 5. ORKIESTRATOR (CREW) - SERCE SYSTEMU
# Tutaj dzieje się magia hierarchii.

business_crew = Crew(
    agents=[analyst, verifier, creative],
    tasks=[task_1_local, task_2_verify, task_3_creative],
    process=Process.hierarchical,
    manager_llm=llm_manager,  # Manager zarządza flow i decyduje kto co robi
    verbose=True,
)

# 6. WYWOŁANIE
print("--- START SYSTEMU AGENTOWEGO ---")
result = business_crew.kickoff()

print("\n\n####################################")
print("RAPORT KOŃCOWY ORKIESTRATORA:")
print(result)
