def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        raw_formatted = []
        
        # 1. Wstępne mapowanie ról
        for m in messages:
            if isinstance(m, SystemChatMessage):
                role = "user"
                content = f"SYSTEM INSTRUCTION: {m.content}"
            elif isinstance(m, AIChatMessage):
                role = "model"
            else:
                role = "user"
                content = m.content
            
            raw_formatted.append({"role": role, "content": content})

        # 2. SCALANIE WIADOMOŚCI (Merging consecutive roles)
        # Gemini nie pozwala na np. ['user', 'user']. Muszą być złączone.
        merged_messages = []
        if raw_formatted:
            current_msg = {
                "role": raw_formatted[0]["role"],
                "parts": [{"text": raw_formatted[0]["content"]}]
            }
            
            for i in range(1, len(raw_formatted)):
                next_role = raw_formatted[i]["role"]
                next_content = raw_formatted[i]["content"]
                
                if next_role == current_msg["role"]:
                    # Jeśli rola ta sama, dopisz do listy parts
                    current_msg["parts"].append({"text": next_content})
                else:
                    # Jeśli rola inna, zapisz poprzednią i zacznij nową
                    merged_messages.append(current_msg)
                    current_msg = {
                        "role": next_role,
                        "parts": [{"text": next_content}]
                    }
            merged_messages.append(current_msg)

        # 3. WYWOŁANIE TWOJEJ BIBLIOTEKI
        print(f"\n[Wrapper] Przekazuję {len(merged_messages)} skonsolidowanych bloków do biblioteki.")
        
        try:
            # Tutaj wywołujesz swoją funkcję z Twojej biblioteki
            # Przekazujemy listę słowników ze strukturą 'role' i 'parts'
            response_text = self.your_library_call(merged_messages)
            
            # Jeśli Twoja biblioteka zwraca obiekt, a nie string, pamiętaj o:
            # response_text = response_text.text 
            
        except Exception as e:
            print(f"❌ [Wrapper] Błąd krytyczny: {e}")
            raise

        return ChatResult(generations=[ChatGeneration(message=AIChatMessage(content=response_text))])

    ---------------------------------------------------
    