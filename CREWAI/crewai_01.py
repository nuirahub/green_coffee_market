from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIChatMessage, BaseMessage, SystemChatMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

# Tutaj zaimportuj swoją bibliotekę
# import moja_biblioteka_gemini as mbG


class GeminiCustomWrapper(BaseChatModel):
    """
    Wrapper łączący CrewAI z Twoją biblioteką obsługującą błędy 429 i fallback modeli.
    """

    model_name: str = "gemini-2.5-pro"
    api_key: str = Field(exclude=True)
    temperature: float = 0.7

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:

        # 1. Konwersja formatu LangChain na format Twojej biblioteki (jeśli potrzebna)
        # Przykład prostej konwersji:
        formatted_messages = []
        for m in messages:
            role = "user"
            if isinstance(m, SystemChatMessage):
                role = "system"
            elif isinstance(m, AIChatMessage):
                role = "assistant"
            formatted_messages.append({"role": role, "content": m.content})

        try:
            # 2. Wywołanie TWOJEJ biblioteki (która ma wbudowane retry i fallback)
            # response_text = mbG.ask(
            #     messages=formatted_messages,
            #     model=self.model_name,
            #     temp=self.temperature
            # )

            # Symulacja odpowiedzi dla przykładu:
            response_text = "To jest odpowiedź wygenerowana przez Twoją bibliotekę."

        except Exception as e:
            # Jeśli Twoja biblioteka mimo 3 prób zawiedzie, CrewAI dostanie jasny komunikat
            raise RuntimeError(f"Twoja biblioteka Gemini nie podołała: {str(e)}")

        # 3. Pakowanie odpowiedzi z powrotem do formatu LangChain
        message = AIChatMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "gemini-custom-wrapper"
---------------------------------------------------

### LLM ###

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Brak zmiennej środowiskowej GEMINI_API_KEY.")

# Używamy stworzonego wrappera zamiast ChatGoogleGenerativeAI
llm = GeminiCustomWrapper(
    model_name="gemini-2.5-pro",
    api_key=api_key,
    temperature=0.7
)


-----------------------

email_writer = Agent(
    role="Specjalista ds. komunikacji email",
    goal="Wygenerować spersonalizowane wiadomości email.",
    backstory="...",
    llm=llm,
    allow_delegation=False,
    max_iter=3,             # Zwiększone
    handle_parsing_errors=True, # Obsługa błędów parsowania
    verbose=True,
)


