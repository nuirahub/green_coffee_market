import time
import logging

# Konfiguracja dedykowanego pliku logów dla błędów API
logger = logging.getLogger("API_Monitor")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("api_limits_monitor.log", encoding="utf-8")
fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
logger.addHandler(fh)
logger.addHandler(logging.StreamHandler()) # Dodatkowo wypisze w konsoli

# ... (wewnątrz Twojej metody _generate we wrapperze) ...

    def _generate(self, messages, **kwargs):
        models_to_try = [self.model_name, self.fallback_model]
        last_exception = None
        
        # Startujemy stoper
        process_start_time = time.time() 

        for attempt_model in models_to_try:
            try:
                logger.info(f"▶️ Start zapytania do modelu: {attempt_model}")
                
                # TU WYWOŁUJESZ SWOJE API
                response_text = self._call_your_library(messages, attempt_model)
                
                # Zatrzymujemy stoper po sukcesie
                elapsed_time = time.time() - process_start_time
                logger.info(f"✅ Sukces ({attempt_model}). Czas całkowity: {elapsed_time:.2f}s")
                
                return ChatResult(generations=[ChatGeneration(message=AIChatMessage(content=response_text))])

            except Exception as e:
                error_msg = str(e).lower()
                time_to_fail = time.time() - process_start_time
                
                # Weryfikacja czy to na pewno błąd 429
                if "429" in error_msg or "exhausted" in error_msg or "rate limit" in error_msg:
                    logger.warning(f"⚠️ BŁĄD 429 (Rate Limit) dla {attempt_model} po {time_to_fail:.2f}s!")
                else:
                    logger.error(f"❌ Inny błąd dla {attempt_model}: {str(e)}")
                
                last_exception = e
                
                # Jeśli mamy jeszcze modele w zapasie, robimy fallback
                if attempt_model != models_to_try[-1]:
                    # Tutaj możesz ustawić ile sekund chcesz odczekać przed zmianą modelu
                    sleep_time = 2 
                    logger.info(f"⏳ Czekam {sleep_time}s przed przełączeniem na model zapasowy...")
                    time.sleep(sleep_time)
                    
                    time_before_fallback = time.time() - process_start_time
                    logger.info(f"🔄 Inicjowanie modelu zapasowego ({self.fallback_model}) w {time_before_fallback:.2f} sekundy od startu procesu.")

        raise last_exception