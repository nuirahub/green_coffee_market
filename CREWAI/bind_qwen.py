# jak się nie uda:
# bound_llm = llm.bind_tools(tools)


import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

# Używamy interfejsu OpenAI do komunikacji z lokalnym Qwenem (Ollama/LM Studio)
# Wymaga: pip install langchain-openai
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

### LLM (Lokalny Qwen via zgodne API OpenAI) ###

# Upewnij się, że wpisujesz tu dokładną nazwę modelu, jaką masz w Ollamie (np. qwen2.5:8b)
MODEL_NAME = "qwen:8b"

llm = ChatOpenAI(
    model=MODEL_NAME,
    base_url="http://localhost:11434/v1",  # Domyślny adres Ollamy. Zmień, jeśli używasz LM Studio.
    api_key="ollama",  # Klucz wymagany przez bibliotekę, ignorowany przez serwer lokalny
    temperature=0.1,  # KLUCZOWE: Niska temperatura dla stabilnego używania narzędzi
    max_retries=3,
)

### TOOLS (z rygorystycznymi schematami Pydantic) ###


class LoadMailingListSchema(BaseModel):
    file_path: str = Field(
        description="Dokładna ścieżka do pliku CSV z listą kontaktów. Zawsze podawaj jako czysty ciąg znaków (string). Przykład: 'data/mailing_list.csv'"
    )


@tool("LoadMailingList", args_schema=LoadMailingListSchema)
def load_mailing_list_tool(file_path: str) -> str:
    """Wczytuje plik csv z kolumnami Name i Email i zwraca JSON z listą kontaktów."""
    try:
        if not os.path.isabs(file_path):
            base_dir = Path(__file__).parent.parent.parent
            file_path = str(base_dir / file_path)

        if not os.path.exists(file_path):
            return json.dumps({"error": f"Plik nie istnieje: {file_path}"})

        df = pd.read_csv(file_path)
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

        contacts_data = []
        for _, row in df.iterrows():
            email = row[email_col]
            if pd.isna(email):
                continue

            contact = {"email": str(email).strip()}
            if name_col and not pd.isna(row[name_col]):
                contact["name"] = str(row[name_col]).strip()
            else:
                contact["name"] = str(email).split("@")[0].split(".")[0].title()

            contacts_data.append(contact)

        return json.dumps(contacts_data, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Błąd podczas wczytywania pliku: {str(e)}"})


class LoadTemplateSchema(BaseModel):
    file_path: str = Field(
        description="Dokładna ścieżka do pliku z szablonem email. Zawsze podawaj jako czysty ciąg znaków (string). Przykład: 'templates/mail_error.html'"
    )


@tool("LoadTemplate", args_schema=LoadTemplateSchema)
def load_template_tool(file_path: str) -> str:
    """Wczytuje template wiadomości email z pliku tekstowego lub HTML."""
    try:
        if not os.path.isabs(file_path):
            base_dir = Path(__file__).parent.parent.parent
            file_path = str(base_dir / file_path)

        if not os.path.exists(file_path):
            return f"Błąd: Plik nie istnieje: {file_path}"

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    except Exception as e:
        return f"Błąd podczas wczytywania template: {str(e)}"


### AGENTS ###

data_loader = Agent(
    role="Specjalista ds. danych mailingowych",
    goal="Wczytać listę mailingową z pliku csv.",
    backstory="Twoim zadaniem jest wywołanie narzędzia LoadMailingList. Podajesz tylko ścieżkę do pliku. Jeśli narzędzie zwróci błąd, popraw ścieżkę.",
    tools=[load_mailing_list_tool],
    llm=llm,
    allow_delegation=False,
    max_iter=3,
    handle_parsing_errors=True,
    verbose=True,
)

template_loader = Agent(
    role="Specjalista ds. template maila",
    goal="Wczytać szablon wiadomości email.",
    backstory="Twoim zadaniem jest wywołanie narzędzia LoadTemplate. Podajesz tylko ścieżkę do pliku. Jeśli narzędzie zwróci błąd, popraw ścieżkę.",
    tools=[load_template_tool],
    llm=llm,
    allow_delegation=False,
    max_iter=3,
    handle_parsing_errors=True,
    verbose=True,
)

email_writer = Agent(
    role="Specjalista ds. komunikacji email",
    goal="Wygenerować spersonalizowane wiadomości email.",
    backstory=(
        "Na podstawie promptu użytkownika generujesz treść maila "
        "i wypełniasz wczytany template dla każdego odbiorcy."
    ),
    llm=llm,
    allow_delegation=False,
    max_iter=3,
    handle_parsing_errors=True,
    verbose=True,
)

manager_agent = Agent(
    role="Project Manager",
    goal="Zarządzać procesem wczytywania kontaktów, szablonu i generowania emaili.",
    backstory="Jesteś konkretny i nadzorujesz pracę zespołu. Jeśli podwładny zgłasza błąd 3 razy, kończysz zadanie raportem o błędzie.",
    llm=llm,
    allow_delegation=True,
    max_iter=3,
    handle_parsing_errors=True,
    verbose=True,
)


### TASKS ###

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
    Zwróć wynik jako czysty JSON.
    """,
    expected_output="JSON z listą kontaktów.",
    agent=data_loader,
)

task_load_template = Task(
    description=f"""
    Wywołaj LoadTemplate z parametrem file_path="{TEMPLATE_FILE}".
    Zwróć dokładną treść template.
    """,
    expected_output="Dokładna treść szablonu z pliku.",
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
    2. Dla każdego kontaktu wstaw jego imię w miejsce {{name}} w szablonie.
    3. Zwróć listę gotowych maili w formacie:
    Email: adres
    Treść: ...
    -----------------
    """,
    expected_output="Lista spersonalizowanych wiadomości email gotowych do wysyłki.",
    agent=email_writer,
    context=[task_load_contacts, task_load_template],
)

main_task = Task(
    description=f"""
    Cel: Wygenerować spersonalizowane wiadomości email.
    
    Kroki do wykonania przez podwładnych:
    1. Wczytaj kontakty z {CONTACTS_FILE}
    2. Wczytaj szablon email z {TEMPLATE_FILE}
    3. Wygeneruj i spersonalizuj maile.
    """,
    expected_output="Lista gotowych, spersonalizowanych wiadomości email.",
)


### CREW ###

logs_dir = BASE_DIR / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

crew = Crew(
    agents=[data_loader, template_loader, email_writer],
    tasks=[main_task],
    process=Process.hierarchical,
    manager_agent=manager_agent,
    max_rpm=10,
    step_callback=lambda x: time.sleep(
        1
    ),  # Delikatne opóźnienie pomaga lokalnym serwerom
    verbose=True,
    output_log_file=str(
        logs_dir / f"crew_process_qwen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    ),
)

### CALL ###

if __name__ == "__main__":
    print("### START CREW (LOCAL QWEN) ###")

    # Podstawowe sprawdzenie plików przed startem
    if not os.path.exists(CONTACTS_FILE):
        print(
            f"⚠️  Ostrzeżenie: Plik {CONTACTS_FILE} nie istnieje! Agent może zgłosić błąd."
        )
    if not os.path.exists(TEMPLATE_FILE):
        print(
            f"⚠️  Ostrzeżenie: Plik {TEMPLATE_FILE} nie istnieje! Agent może zgłosić błąd."
        )

    try:
        result = crew.kickoff()
        print("\n" + "=" * 50)
        print("### WYNIK CREW ###")
        print("=" * 50)
        print(result)

    except Exception as e:
        print(f"\n❌ Błąd podczas wykonywania crew: {type(e).__name__}")
        print(f"Szczegóły: {str(e)}")
