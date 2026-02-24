import json
import operator
import os
from pathlib import Path
from typing import Annotated, Literal, Sequence, TypedDict

import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

# ==========================================
# 1. KONFIGURACJA LLM (Lokalny Qwen3)
# ==========================================

# Używamy interfejsu OpenAI dla natywnego wsparcia wywoływania narzędzi
llm = ChatOpenAI(
    model="qwen3:8b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0.1,  # Niska temperatura to konieczność przy narzędziach
    max_retries=3,
)

# ==========================================
# 2. NARZĘDZIA (Ścisłe schematy Pydantic)
# ==========================================


class LoadMailingListSchema(BaseModel):
    file_path: str = Field(
        description="Ścieżka do pliku CSV. Zawsze używaj: 'data/mailing_list.csv'"
    )


@tool("LoadMailingList", args_schema=LoadMailingListSchema)
def load_mailing_list_tool(file_path: str) -> str:
    """Wczytuje listę kontaktów z pliku CSV i zwraca JSON."""
    # (Uproszczona logika dla przykładu, wstaw tu swoją pełną ścieżkę z poprzedniego kodu)
    try:
        base_dir = Path(__file__).parent
        full_path = (
            str(base_dir / file_path) if not os.path.isabs(file_path) else file_path
        )

        if not os.path.exists(full_path):
            return json.dumps({"error": f"Plik {full_path} nie istnieje."})

        df = pd.read_csv(full_path)
        contacts = [
            {"email": row.get("Email", ""), "name": row.get("Name", "Klient")}
            for _, row in df.iterrows()
        ]
        return json.dumps(contacts, ensure_ascii=False)
    except Exception as e:
        return f"Błąd: {str(e)}"


class LoadTemplateSchema(BaseModel):
    file_path: str = Field(
        description="Ścieżka do szablonu. Zawsze używaj: 'templates/mail_error.html'"
    )


@tool("LoadTemplate", args_schema=LoadTemplateSchema)
def load_template_tool(file_path: str) -> str:
    """Wczytuje szablon email."""
    try:
        base_dir = Path(__file__).parent
        full_path = (
            str(base_dir / file_path) if not os.path.isabs(file_path) else file_path
        )
        if not os.path.exists(full_path):
            return f"Błąd: Plik {full_path} nie istnieje."
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Błąd: {str(e)}"


tools = [load_mailing_list_tool, load_template_tool]


# ==========================================
# 3. DEFINICJA STANU (STATE)
# ==========================================
# Stan przechowuje historię wiadomości wymienianych między agentami i orkiestratorem
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


# ==========================================
# 4. ORKIESTRATOR (SUPERVISOR)
# ==========================================
# Orkiestrator decyduje, kto ma wykonać następny krok.
members = ["Data_Agent", "Email_Agent"]
system_prompt = (
    "Jesteś Orkiestratorem (Supervisor). Zarządzasz następującymi pracownikami: {members}.\n"
    "Twoje zadanie to wygenerowanie emaili. Aby to zrobić:\n"
    "1. Najpierw wyślij zadanie do 'Data_Agent', aby wczytał kontakty (LoadMailingList) i szablon (LoadTemplate).\n"
    "2. Kiedy Data_Agent wczyta pliki, wyślij zadanie do 'Email_Agent', aby napisał maile.\n"
    "3. Kiedy maile będą gotowe, odpowiedz 'FINISH'.\n"
    "Zawsze wybieraj jednego z pracowników lub FINISH."
)

options = ["FINISH"] + members


class RouteResponse(BaseModel):
    next: Literal[(*options,)]


supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Biorąc pod uwagę powyższą rozmowę, kto powinien działać teraz? Wybierz jedną z opcji: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

# Zmuszamy Qwena do zwrócenia tylko sztywnego JSONa z routingiem
supervisor_chain = supervisor_prompt | llm.with_structured_output(RouteResponse)

# ==========================================
# 5. AGENCI WĘZŁOWI (WORKERS)
# ==========================================

# 5.1. Data Agent (Używa narzędzi z użyciem wbudowanego create_react_agent z LangGraph)
data_agent_prompt = "Jesteś asystentem ds. danych. Użyj narzędzi, aby wczytać plik CSV oraz plik z szablonem. Gdy to zrobisz, poinformuj o sukcesie i wypisz dane."
data_agent = create_react_agent(llm, tools=tools, state_modifier=data_agent_prompt)


def data_node(state: AgentState):
    print("--- 🔍 DATA AGENT PRACUJE ---")
    result = data_agent.invoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="Data_Agent")
        ]
    }


# 5.2. Email Agent (Nie ma narzędzi, tylko przetwarza kontekst)
def email_node(state: AgentState):
    print("--- ✉️ EMAIL AGENT PRACUJE ---")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Jesteś copywriterem. W poprzednich wiadomościach otrzymasz listę kontaktów JSON oraz szablon email. "
                "Twoim zadaniem jest podstawienie imion z JSONa w miejsce {name} w szablonie i wygenerowanie gotowych wiadomości. "
                "Zwróć wynik jako gotowe wiadomości.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | llm
    result = chain.invoke(state)
    return {"messages": [HumanMessage(content=result.content, name="Email_Agent")]}


# 5.3. Węzeł Orkiestratora
def supervisor_node(state: AgentState):
    print("--- 🧠 ORKIESTRATOR MYŚLI ---")
    decision = supervisor_chain.invoke(state)
    print(f"    Orkiestrator kieruje do: {decision.next}")
    return {"next": decision.next}


# ==========================================
# 6. BUDOWA GRAFU (LANGGRAPH)
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Data_Agent", data_node)
workflow.add_node("Email_Agent", email_node)

# Zdefiniowanie ścieżek
workflow.add_edge("Data_Agent", "Supervisor")
workflow.add_edge("Email_Agent", "Supervisor")

# Logika rutingu z Orkiestratora
workflow.add_conditional_edges(
    "Supervisor",
    lambda state: state["next"],
    {"Data_Agent": "Data_Agent", "Email_Agent": "Email_Agent", "FINISH": END},
)

workflow.add_edge(START, "Supervisor")

# Kompilacja grafu
app = workflow.compile()

# ==========================================
# 7. URUCHOMIENIE
# ==========================================
if __name__ == "__main__":
    print("### START LANGGRAPH (QWEN3:8B) ###\n")

    # Inicjalne polecenie
    inputs = {
        "messages": [
            HumanMessage(
                content="Przygotuj maile. Użyj plików: 'data/mailing_list.csv' oraz 'templates/mail_error.html'. Wczytaj je, a potem wygeneruj wiadomości."
            )
        ]
    }

    # Przejście przez graf
    for step in app.stream(inputs, stream_mode="values"):
        last_message = step["messages"][-1]
        # Wyświetlamy tylko nowe wiadomości
        if "name" in last_message.additional_kwargs or hasattr(last_message, "name"):
            print(
                f"\n[Wyjście od {last_message.name}]:\n{last_message.content[:200]}...\n"
            )
            print("-" * 30)

    print("\n✅ PROCES ZAKOŃCZONY.")
