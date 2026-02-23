import json
import os
from pathlib import Path

import pandas as pd
from crewai.tools import tool

# pip install langchain-google-genai
from langchain_google_genai import ChatGoogleGenerativeAI

### TOOLS ###


@tool("LoadMailingList")
def load_mailing_list_tool(file_path: str) -> str:
    """Wczytuje plik csv z kolumnami Name i Email i zwraca JSON."""

    try:
        # Konwertuj względną ścieżkę na absolutną jeśli potrzeba

        if not os.path.isabs(file_path):
            base_dir = Path(__file__).parent.parent.parent

            file_path = str(base_dir / file_path)

        if not os.path.exists(file_path):
            return json.dumps({"error": f"Plik nie istnieje: {file_path}"})

        df = pd.read_csv(file_path)

        # Sprawdź dostępność kolumn (case-insensitive)

        df.columns = df.columns.str.strip()

        email_col = None

        name_col = None

        for col in df.columns:
            if col.lower() == "email":
                email_col = col

            if col.lower() == "name":
                name_col = col

        if email_col is None:
            return json.dumps({"error": "Nie znaleziono kolumny 'Email' w pliku CSV"})

        # Przygotuj dane - użyj Name jeśli istnieje, w przeciwnym razie użyj Email jako fallback

        contacts_data = []

        for _, row in df.iterrows():
            email = row[email_col]

            if pd.isna(email):
                continue

            contact = {"email": str(email).strip()}

            if name_col and not pd.isna(row[name_col]):
                contact["name"] = str(row[name_col]).strip()

            else:
                # Jeśli nie ma kolumny Name, użyj części przed @ z emaila jako imię

                contact["name"] = str(email).split("@")[0].split(".")[0].title()

            contacts_data.append(contact)

        return json.dumps(contacts_data, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Błąd podczas wczytywania pliku: {str(e)}"})


@tool("LoadTemplate")
def load_template_tool(file_path: str) -> str:
    """Wczytuje template wiadomości email z pliku txt."""

    try:
        # Konwertuj względną ścieżkę na absolutną jeśli potrzeba

        if not os.path.isabs(file_path):
            base_dir = Path(__file__).parent.parent.parent

            file_path = str(base_dir / file_path)

        if not os.path.exists(file_path):
            return f"Błąd: Plik nie istnieje: {file_path}"

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    except Exception as e:
        return f"Błąd podczas wczytywania template: {str(e)}"


### LLM ###

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "Brak zmiennej środowiskowej GEMINI_API_KEY. Ustaw ją w pliku .env"
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=api_key,
    temperature=0.7,
)


### AGENTS ###

from crewai import Agent

data_loader = Agent(
    role="Specjalista ds. danych mailingowych",
    goal="Wczytać listę mailingową z pliku csv.",
    backstory="Twoim zadaniem jest wywołanie narzędzia LoadMailingList do wczytania pliku CSV z listą kontaktów i zwrócenie danych w formacie JSON.",
    tools=[load_mailing_list_tool],
    llm=llm,
    allow_delegation=False,
    max_iter=1,
)


template_loader = Agent(
    role="Specjalista ds. template maila",
    goal="Wczytać szablon wiadomości email.",
    backstory="Twoim zadaniem jest wywołanie narzędzia LoadTemplate do wczytania pliku z szablonem wiadomości email i zwrócenie jego treści.",
    tools=[load_template_tool],
    llm=llm,
    allow_delegation=False,
    max_iter=1,
)


email_writer = Agent(
    role="Specjalista ds. komunikacji email",
    goal="Wygenerować spersonalizowane wiadomości email.",
    backstory=(
        "Na podstawie promptu użytkownika generujesz treść maila "
        "i wypełniasz template dla każdego odbiorcy."
    ),
    llm=llm,
    allow_delegation=False,
    max_iter=1,
    verbose=True,
)


manager_agent = Agent(
    role="Project Manager",
    goal="Manage the process of loading contacts and template and generating emails. ",
    backstory="You are specific. If a subordinate reports an error 3 times, you end the task with a report of the error. You are the manager of the crew.",
    max_iter=1,
    # max_rpm=3,
    allow_delegation=True,
    # verbose=True,
    llm=llm,
)


### TASKS ###

from crewai import Task

# Ścieżki do plików - można nadpisać zmiennymi środowiskowymi lub użyć domyślnych

BASE_DIR = Path(__file__).parent.parent.parent

CONTACTS_FILE = os.getenv(
    "MAIL_CONTACTS_FILE",
    str(BASE_DIR / "data" / "green_coffe_market" / "mailing_list.csv"),
)

TEMPLATE_FILE = os.getenv(
    "MAIL_TEMPLATE_FILE", str(BASE_DIR / "templates" / "mail_error.html")
)


user_prompt = (
    "Napisz profesjonalnego maila informującego o nowej ofercie szkoleniowej AI."
)


task_load_contacts = Task(
    description=f"""

    Wywołaj LoadMailingList z parametrem file_path="{CONTACTS_FILE}".

    Zwróć wynik jako JSON.

    """,
    expected_output="JSON lista kontaktów.",
    agent=data_loader,
)


task_load_template = Task(
    description=f"""

    Wywołaj LoadTemplate z parametrem file_path="{TEMPLATE_FILE}".

    Zwróć dokładną treść template.

    """,
    expected_output="Treść template.",
    agent=template_loader,
)


task_generate_emails = Task(
    description=f"""

    Oto lista kontaktów:

    {{task_load_contacts}}

 

    Oto template:

    {{task_load_template}}

 

    Prompt użytkownika:

    {user_prompt}

 

    1. Wygeneruj treść maila na podstawie promptu.

    2. Dla każdego kontaktu wstaw imię w miejsce {{name}}.

    3. Zwróć listę gotowych maili w formacie:

 

    Email: adres

    Treść:

    ...

    -----------------

    """,
    expected_output="Lista gotowych wiadomości email w formacie: Email: adres Treść: ... ",
    agent=email_writer,
    context=[task_load_contacts, task_load_template],
)


main_task = Task(
    description=f"""

    Goal: Generate personalized emails.

 

    Steps required:

    1. Load contacts from {CONTACTS_FILE}

    2. Load email template from {TEMPLATE_FILE}

    3. Generate email content based on user prompt:

       {user_prompt}

    4. Personalize emails for each contact.

    5. Send emails to each contact.

    6. Deliver final list of ready-to-send emails.

    """,
    expected_output="List of personalized emails ready to send. Send emails to each contact.",
)


### CREW ###

from datetime import datetime, time

from crewai import Crew, Process

# Utwórz katalog logs jeśli nie istnieje

logs_dir = BASE_DIR / "logs"

logs_dir.mkdir(exist_ok=True)


crew = Crew(
    agents=[data_loader, template_loader, email_writer],
    tasks=[main_task],
    process=Process.hierarchical,
    manager_agent=manager_agent,
    max_rpm=10,
    step_callback=lambda x: time.sleep(2),
    verbose=True,
    output_log_file=str(
        logs_dir / f"crew_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    ),
    tracing=True,
)


### CALL ###

if __name__ == "__main__":
    print("### START CREW ###")

    try:
        # Sprawdź czy pliki istnieją przed uruchomieniem

        if not os.path.exists(CONTACTS_FILE):
            print(f"⚠️  Ostrzeżenie: Plik {CONTACTS_FILE} nie istnieje!")

        if not os.path.exists(TEMPLATE_FILE):
            print(f"⚠️  Ostrzeżenie: Plik {TEMPLATE_FILE} nie istnieje!")

        result = crew.kickoff()

        print("\n" + "=" * 50)

        print("### WYNIK CREW ###")

        print("=" * 50)

        print(result)

    except Exception as e:
        print(f"\n❌ Błąd podczas wykonywania crew: {type(e).__name__}")

        print(f"Szczegóły: {str(e)}")

        raise
