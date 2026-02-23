from typing import Any, List, Optional, Dict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIChatMessage, HumanChatMessage, SystemChatMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field
import logging

# Konfiguracja logowania, abyś widział co robi wrapper
logger = logging.getLogger("GeminiWrapper")

class GeminiCustomWrapper(BaseChatModel):
    """
    Wrapper integrujący Twoją bibliotekę z CrewAI.
    Obsługuje mapowanie ról i przekazuje logikę retry/fallback do Twojego kodu.
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
        
        # 1. MAPOWANIE RÓL
        # CrewAI wysyła listę obiektów LangChain. Musimy to zamienić na listę słowników.
        formatted_messages = []
        for m in messages:
            if isinstance(m, SystemChatMessage):
                role = "system"
            elif isinstance(m, AIChatMessage):
                role = "assistant"
            else:
                role = "user"
            
            formatted_messages.append({
                "role": role, 
                "parts": [{"text": m.content}]  # Format specyficzny dla Gemini API
            })

        print(f"\n[Wrapper] Wysyłam zapytanie do Twojej biblioteki (Model: {self.model_name})...")

        try:
            # 2. WYWOŁANIE TWOJEJ BIBLIOTEKI
            # Tutaj wstawiasz realne wywołanie swojej funkcji:
            # response_text = TwojaBiblioteka.generate(
            #     messages=formatted_messages,
            #     temperature=self.temperature,
            #     model=self.model_name
            # )
            
            # Placeholder dla testu:
            response_text = "Treść wygenerowana przez Twoją bibliotekę po ewentualnych retry."

        except Exception as e:
            print(f"❌ [Wrapper] Krytyczny błąd Twojej biblioteki po wszystkich próbach: {e}")
            raise

        # 3. ZWRACANIE WYNIKU DO CREWAI
        message = AIChatMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "gemini-custom-wrapper"





---------------------------------------------------
### LLM ###

# Inicjalizacja Twojego wrappera
llm = GeminiCustomWrapper(
    api_key=api_key,
    model_name="gemini-2.0-flash", # Możesz tu podać domyślny model
    temperature=0.7
)

### AGENTS ###

# Zmieniamy max_iter na 3, aby agent mógł ponowić próbę, 
# jeśli Twoja biblioteka zwróci błąd lub błąd parsowania.

data_loader = Agent(
    role="Specjalista ds. danych mailingowych",
    goal="Wczytać listę mailingową z pliku csv.",
    backstory="...",
    tools=[load_mailing_list_tool],
    llm=llm,
    allow_delegation=False,
    max_iter=3,             # Zwiększone z 1
    verbose=True,           # Włączone dla widoczności
    handle_parsing_errors=True
)

# Powtórz analogicznie dla template_loader i email_writer





---------------------------------------------------


def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        formatted_messages = []
        
        for m in messages:
            # Mapowanie ról LangChain na role akceptowane przez Gemini API
            if isinstance(m, SystemChatMessage):
                # System prompt w Gemini najlepiej podać jako pierwszą wiadomość 'user' 
                # lub użyć 'system_instruction' (jeśli Twoja biblioteka to wspiera)
                role = "user" 
                content = f"SYSTEM INSTRUCTION: {m.content}"
            elif isinstance(m, AIChatMessage):
                role = "model" # Gemini używa 'model' zamiast 'assistant'
                content = m.content
            else:
                role = "user"
                content = m.content
            
            # Budowa struktury 'parts'
            formatted_messages.append({
                "role": role,
                "parts": [{"text": content}]
            })

        # Logowanie przed wysłaniem
        print(f"\n[Wrapper] Przekazuję {len(formatted_messages)} wiadomości do biblioteki zewnętrznej.")

        try:
            # WYWOŁANIE TWOJEJ BIBLIOTEKI
            # Zakładamy, że Twoja funkcja przyjmuje listę słowników z kluczem 'parts'
            response_text = self.your_library_call(formatted_messages)
            
        except Exception as e:
            # Tutaj Twoja biblioteka już zrobiła 3 retries i ewentualny fallback modelu
            print(f"❌ [Wrapper] Wszystkie próby zawiodły. Ostatni błąd: {e}")
            raise

        message = AIChatMessage(content=response_text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def your_library_call(self, messages):
        """
        Miejsce na wywołanie Twojej logiki.
        Możesz tu dodać printy, które pokażą, który model aktualnie odpowiada.
        """
        # return MojaBiblioteka.ask(messages, temperature=self.temperature)
        pass