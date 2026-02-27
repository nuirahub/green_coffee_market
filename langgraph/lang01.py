from typing import List, TypedDict

from jinja2 import Template
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

# --- KONFIGURACJA ---
# Upewnij się, że masz ustawiony klucz API Gemini w systemie
# os.environ["GOOGLE_API_KEY"] = "TWOJ_KLUCZ"

local_llm = ChatOllama(model="qwen2.5:8b")  # Lokalny Qwen przez Ollama
remote_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")  # Gemini (Free Tier)
search_tool = DuckDuckGoSearchRun()


# --- DEFINICJA STANU ---
class AgentState(TypedDict):
    user_prompt: str
    recipients: List[dict]
    search_queries: List[str]
    raw_content: str
    report: str
    final_emails: List[dict]


# --- WĘZŁY (NODES) ---


def planner_node(state: AgentState):
    """Qwen generuje frazy do wyszukiwarki."""
    prompt = f"Na podstawie prośby: '{state['user_prompt']}', przygotuj 3 konkretne zapytania do Google po angielsku i polsku, oddzielone przecinkami."
    response = local_llm.invoke(prompt)
    queries = [q.strip() for q in response.content.split(",")]
    return {"search_queries": queries}


def search_and_scrape_node(state: AgentState):
    """DuckDuckGo + Jina Reader (100% Free)."""
    combined_text = ""
    for query in state["search_queries"][:2]:  # Ograniczamy do 2 zapytań dla szybkości
        print(f"🔎 Szukam: {query}")
        results = search_tool.run(query)  # Pobiera snippety
        # Opcjonalnie: Tutaj można by wyciągnąć URL i użyć r.jina.ai/URL
        # Dla uproszczenia i stabilności używamy rozszerzonych snippetów z DDG
        combined_text += f"\nWyniki dla {query}:\n{results}"

    return {"raw_content": combined_text}


def analyzer_node(state: AgentState):
    """Gemini analizuje dużą ilość danych i tworzy raport."""
    prompt = f"""
    Jesteś analitykiem. Na podstawie poniższych danych z sieci:
    {state["raw_content"]}
    
    Stwórz strukturalny raport dotyczący: {state["user_prompt"]}.
    Podziel go na kategorie: 'Najważniejsze', 'Trendy', 'Prognozy'.
    Użyj czystego tekstu.
    """
    response = remote_llm.invoke(prompt)
    return {"report": response.content}


def mailer_node(state: AgentState):
    """Qwen + Jinja2 składają maile dla każdego odbiorcy."""
    email_template = """
    Cześć {{ name }},
    
    Oto najnowszy raport przygotowany dla branży {{ industry }}:
    
    {{ report }}
    
    Pozdrawiamy,
    Twój AI Agent
    """
    tm = Template(email_template)
    emails = []

    for contact in state["recipients"]:
        body = tm.render(
            name=contact["name"], industry=contact["industry"], report=state["report"]
        )
        emails.append({"to": contact["email"], "content": body})
        print(f"📧 Wygenerowano maila dla: {contact['email']}")

    return {"final_emails": emails}


# --- KONSTRUKCJA GRAFU ---

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("searcher", search_and_scrape_node)
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("mailer", mailer_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "searcher")
workflow.add_edge("searcher", "analyzer")
workflow.add_edge("analyzer", "mailer")
workflow.add_edge("mailer", END)

app = workflow.compile()

# --- URUCHOMIENIE ---
if __name__ == "__main__":
    inputs = {
        "user_prompt": "Najnowsze trendy w automatyzacji procesów biznesowych AI 2026",
        "recipients": [
            {"name": "Jan", "email": "jan@firma.pl", "industry": "Logistyka"},
            {"name": "Anna", "email": "anna@tech.pl", "industry": "E-commerce"},
        ],
    }

    result = app.invoke(inputs)
    print("\n--- PROCES ZAKOŃCZONY ---")
    # Tutaj możesz dodać funkcję faktycznego wysyłania (smtplib)
