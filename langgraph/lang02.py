1. Rozszerzenie Stanu (State)
Musisz dodać do stanu licznik prób (żeby uniknąć nieskończonej pętli) oraz pole na informację zwrotną od modelu.

Python
class AgentState(TypedDict):
    user_prompt: str
    search_queries: List[str]
    raw_content: str
    is_satisfactory: bool  # Czy dane są poprawne?
    feedback: str         # Co poprawić w wyszukiwaniu?
    iteration_count: int  # Licznik pętli
2. Węzeł Oceny (Grade Node)
Tutaj Twój lokalny Qwen (lub Gemini) sprawdza, czy to, co znalazł Jina i DuckDuckGo, ma sens.

Python
def grader_node(state: AgentState):
    """Qwen ocenia, czy dane są wystarczające."""
    prompt = f"""
    Użytkownik szuka: {state['user_prompt']}
    Znalezione dane: {state['raw_content'][:2000]} # Analiza fragmentu
    
    Czy te dane pozwalają na stworzenie rzetelnego raportu? 
    Odpowiedz w formacie JSON: 
    {{"is_satisfactory": true/false, "feedback": "czego brakuje"}}
    """
    # Używamy Qwena z formatem JSON
    response = local_llm.invoke(prompt)
    data = json.loads(response.content)
    
    return {
        "is_satisfactory": data["is_satisfactory"],
        "feedback": data["feedback"],
        "iteration_count": state["iteration_count"] + 1
    }
3. Logika sterowania (Conditional Edge)
Teraz najważniejsze: definiujemy funkcję, która decyduje o kierunku przepływu w grafie.

Python
def decide_to_continue(state: AgentState):
    # Jeśli dane są OK lub próbowaliśmy już 3 razy - kończymy i robimy raport
    if state["is_satisfactory"] or state["iteration_count"] >= 3:
        return "generate_report"
    # W przeciwnym razie - wracamy do planowania nowych zapytań
    return "replan_search"
4. Definicja Grafu z Pętlą
Wizualnie wygląda to tak: Planner -> Searcher -> Grader -> (Pętla do Planner lub wyjście do Analyzer).

Python
workflow = StateGraph(AgentState)

# Dodanie węzłów
workflow.add_node("planner", planner_node)
workflow.add_node("searcher", search_and_scrape_node)
workflow.add_node("grader", grader_node)
workflow.add_node("analyzer", analyzer_node) # To nasz "generate_report"

# Połączenia
workflow.set_entry_point("planner")
workflow.add_edge("planner", "searcher")
workflow.add_edge("searcher", "grader")

# Krawędź warunkowa po ocenie
workflow.add_conditional_edges(
    "grader",
    decide_to_continue,
    {
        "replan_search": "planner",   # WRACAMY (Pętla)
        "generate_report": "analyzer" # IDZIEMY DALEJ
    }
)

workflow.add_edge("analyzer", END)
app = workflow.compile()
Jak to działa w praktyce?
Planner dostaje feedback ze stanu (jeśli to kolejna iteracja). Jeśli feedback mówi "brak danych o cenach", Qwen w nowym kroku wygeneruje zapytania specyficzne pod ceny.

Searcher pobiera nowe dane i dopisuje je do raw_content.

Grader widzi już więcej danych i może powiedzieć is_satisfactory: true.

Bezpiecznik: iteration_count gwarantuje, że nie utkniesz w pętli na zawsze (i nie spalisz prądu/tokenów), jeśli internet milczy na dany temat.

Dlaczego to podejście jest lepsze niż CrewAI?
W CrewAI agenci "kręcą się" w sposób niekontrolowany. Tutaj Ty kontrolujesz warunek wyjścia. Możesz ustawić, że grader ma być bardzo surowy dla Gemini, albo wręcz przeciwnie – bardzo pobłażliwy, żeby oszczędzać czas.