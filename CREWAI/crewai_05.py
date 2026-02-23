import time
from typing import Any, List, Optional, Dict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIChatMessage, SystemChatMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field

class GeminiCustomWrapper(BaseChatModel):
    """
    Wrapper z wbudowanym mapowaniem ról, scalaniem wiadomości, 
    ponawianiem prób (retry) oraz automatyczną zmianą modelu (fallback) w przypadku błędów 429.
    """
    model_name: str = "gemini-2.5-pro"
    # Lista modeli zapasowych, używanych gdy główny model rzuca błędami limitów
    fallback_models: List[str] = Field(default_factory=lambda: ["gemini-2.5-flash", "gemini-2.0-flash"])
    api_key: str = Field(exclude=True)
    temperature: float = 0.7
    max_retries_per_model: int = 3

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        # 1. MAPOWANIE RÓL
        raw_formatted = []
        for m in messages:
            if isinstance(m, SystemChatMessage):
                role = "user"
                content = f"SYSTEM INSTRUCTION: {m.content}"
            elif isinstance(m, AIChatMessage):
                role = "model"
                content = m.content
            else:
                role = "user"
                content = m.content
            raw_formatted.append({"role": role, "content": content})

        # 2. SCALANIE WIADOMOŚCI (Merging)
        merged_messages = []
        if raw_formatted:
            current_msg = {"role": raw_formatted[0]["role"], "parts": [{"text": raw_formatted[0]["content"]}]}
            for i in range(1, len(raw_formatted)):
                next_role = raw_formatted[i]["role"]
                next_content = raw_formatted[i]["content"]
                if next_role == current_msg["role"]:
                    current_msg["parts"].append({"text": next_content})
                else:
                    merged_messages.append(current_msg)
                    current_msg = {"role": next_role, "parts": [{"text": next_content}]}
            merged_messages.append(current_msg)

        # 3. LOGIKA RETRY & FALLBACK
        models_to_try = [self.model_name] + self.fallback_models
        
        for current_model in models_to_try:
            for attempt in range(self.max_retries_per_model):
                try:
                    print(f"\n[Wrapper] Zapytanie do: {current_model} (Próba {attempt + 1}/{self.max_retries_per_model})")
                    
                    # ---------------------------------------------------------
                    # TUTAJ WYWOŁAJ SWOJĄ BIBLIOTEKĘ Z AKTUALNYM 'current_model'
                    # ---------------------------------------------------------
                    # Przykład: 
                    # response_text = MojaBiblioteka.ask(
                    #     messages=merged_messages, 
                    #     model=current_model, 
                    #     temperature=self.temperature
                    # )
                    
                    # Symulacja poprawnej odpowiedzi:
                    response_text = f"Odpowiedź wygenerowana przez {current_model}"
                    
                    # Jeśli się udało, zwracamy wynik i kończymy
                    message = AIChatMessage(content=response_text)
                    return ChatResult(generations=[ChatGeneration(message=message)])
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    print(f"⚠️ [Wrapper] Błąd ({current_model}): {e}")
                    
                    # Sprawdzenie czy to błąd związany z limitami (429, quota, rate limit)
                    if "429" in error_msg or "quota" in error_msg or "rate" in error_msg:
                        if attempt < self.max_retries_per_model - 1:
                            print("[Wrapper] Przekroczono limit. Czekam 5 sekund przed ponowieniem...")
                            time.sleep(5)
                            continue  # Ponów próbę dla tego samego modelu
                        else:
                            print(f"🔄 [Wrapper] Wyczerpano limity dla {current_model}. Przełączam na model zapasowy...")
                            break  # Przerwij wewnętrzną pętlę, przejdź do następnego modelu
                    else:
                        # Inny błąd API (np. 500 lub zła struktura) - ponawiamy szybciej
                        if attempt < self.max_retries_per_model - 1:
                            time.sleep(2)
                            continue
                        # Jeśli to nie błąd 429 i skończyły się próby, rzucamy wyjątek wyżej
                        raise e 
                        
        # Jeśli pętla przeszła przez wszystkie modele i nic nie zwróciła
        raise RuntimeError("❌ [Wrapper] Krytyczny błąd: Wszystkie modele fallbackowe zawiodły lub wyczerpano limity API.")

    @property
    def _llm_type(self) -> str:
        return "gemini-custom-fallback-wrapper"



    ---------------------------------------------------


    import logging
import time
from typing import Any, List, Optional, Dict
from pydantic import Field, PrivateAttr

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIChatMessage, SystemChatMessage
from langchain_core.outputs import ChatResult, ChatGeneration

# Konfiguracja loggera, abyś widział proces w konsoli
logger = logging.getLogger("GeminiWrapper")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - [GeminiWrapper] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class GeminiCustomWrapper(BaseChatModel):
    """
    Wrapper dla CrewAI obsługujący:
    - Mapowanie ról (System -> User, Assistant -> Model)
    - Scalanie (Merging) wiadomości o tej samej roli
    - Automatyczny Fallback modeli przy błędach (np. 429)
    """
    
    # Konfiguracja modeli
    model_name: str = "gemini-2.0-pro-exp-02-11"  # Główny model
    fallback_model: str = "gemini-2.0-flash"      # Model zapasowy
    
    api_key: str = Field(exclude=True)
    temperature: float = 0.7
    max_retries: int = 3
    
    # Wewnętrzne flagi
    _is_fallback_active: bool = PrivateAttr(default=False)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        # 1. PRZYGOTOWANIE STRUKTURY DANYCH (MAPPING & MERGING)
        gemini_messages = self._prepare_messages(messages)
        
        # 2. LOGIKA WYWOŁANIA Z FALLBACKIEM
        # Definiujemy kolejność: Najpierw główny, potem fallback
        models_to_try = [self.model_name, self.fallback_model]
        
        last_exception = None
        
        for attempt_model in models_to_try:
            try:
                logger.info(f"Próba generowania. Model: {attempt_model}. Ilość bloków: {len(gemini_messages)}")
                
                # --- TU JEST MIEJSCE NA TWOJĄ BIBLIOTEKĘ ---
                response_text = self._call_your_library(
                    messages=gemini_messages,
                    model=attempt_model
                )
                # -------------------------------------------
                
                # Jeśli się udało, zwracamy wynik
                message = AIChatMessage(content=response_text)
                return ChatResult(generations=[ChatGeneration(message=message)])

            except Exception as e:
                logger.warning(f"Błąd modelu {attempt_model}: {str(e)}")
                last_exception = e
                # Jeśli to był ostatni model na liście, nie mamy już opcji
                if attempt_model == models_to_try[-1]:
                    logger.error("Wszystkie modele zawiodły.")
                else:
                    logger.info(f"Przełączam na model zapasowy: {self.fallback_model}...")
                    time.sleep(2) # Krótka pauza przed zmianą modelu

        # Jeśli pętla się skończyła i nic nie zwrócono:
        raise last_exception

    def _prepare_messages(self, messages: List[BaseMessage]) -> List[Dict]:
        """Konwertuje wiadomości LangChain na format Gemini z łączeniem ról."""
        
        raw_formatted = []
        
        # Krok A: Wstępne mapowanie
        for m in messages:
            if isinstance(m, SystemChatMessage):
                role = "user"
                content = f"SYSTEM INSTRUCTION: {m.content}"
            elif isinstance(m, AIChatMessage):
                role = "model"
                content = m.content
            else:
                role = "user"
                content = m.content
            
            # Pomijamy puste wiadomości, które czasem generuje LangChain
            if content:
                raw_formatted.append({"role": role, "text": content})

        if not raw_formatted:
            return []

        # Krok B: Scalanie (Merging)
        merged_messages = []
        
        current_msg = {
            "role": raw_formatted[0]["role"],
            "parts": [{"text": raw_formatted[0]["text"]}]
        }

        for i in range(1, len(raw_formatted)):
            next_role = raw_formatted[i]["role"]
            next_text = raw_formatted[i]["text"]

            if next_role == current_msg["role"]:
                # Ta sama rola -> doklejamy do parts
                current_msg["parts"].append({"text": next_text})
            else:
                # Inna rola -> zamykamy obecną, otwieramy nową
                merged_messages.append(current_msg)
                current_msg = {
                    "role": next_role,
                    "parts": [{"text": next_text}]
                }
        
        merged_messages.append(current_msg)
        return merged_messages

    def _call_your_library(self, messages, model):
        """
        Symulacja wywołania Twojej biblioteki. 
        Zastąp to właściwym importem i wywołaniem.
        """
        # Przykład integracji:
        # from twoja_lib import GeminiClient
        # client = GeminiClient(api_key=self.api_key)
        # return client.generate_content(model=model, contents=messages)
        
        # PONIŻEJ TYLKO SYMULACJA DO TESTÓW (usuń to w produkcji)
        if "error" in messages[-1]['parts'][0]['text'].lower():
             raise ValueError("Symulowany błąd 429 Resource Exhausted")
        
        return f"To jest odpowiedź z modelu {model}. Otrzymałem {len(messages)} wiadomości."

    @property
    def _llm_type(self) -> str:
        return "gemini-custom-fallback-wrapper"


    ### LLM ###

api_key = os.getenv("GEMINI_API_KEY")

# Inicjalizacja Twojego Custom Wrappera
llm = GeminiCustomWrapper(
    api_key=api_key,
    model_name="gemini-2.0-pro-exp-02-11", # Główny, potężny model
    fallback_model="gemini-2.0-flash",      # Szybki, tani, rzadziej łapie limity
    temperature=0.7
)


email_writer = Agent(
    role="Specjalista ds. komunikacji email",
    # ... reszta parametrów ...
    llm=llm,
    max_iter=3,                # Ważne: Daje szansę na poprawę
    handle_parsing_errors=True, # Ważne: Agent sam spróbuje naprawić JSON
    verbose=True
)
