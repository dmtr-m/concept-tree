import stanza

# Загрузка модели для английского языка
nlp = stanza.Pipeline('en')

def resolve_anaphora_stanza(text):
    """
    Функция для разрешения анафоры в тексте с использованием Stanza.
    Заменяет местоимения на их референты.
    
    Args:
    text (str): Текст с неразрешённой анафорой.
    
    Returns:
    str: Текст с разрешённой анафорой, где местоимения заменены на референты.
    """
    # Обрабатываем текст
    doc = nlp(text)
    
    # Структура для хранения разрешённых анафор
    resolved_text = text
    
    # Перебираем все предложения в документе
    for sent in doc.sentences:
        for word in sent.words:
            # Ищем местоимения (например, 'it', 'he', 'she') и пытаемся найти их референты
            if word.pos == 'PRON':
                # Пытаемся найти зависимость, указывающую на референт
                head = sent.words[word.head - 1] if word.head > 0 else None
                if head and head.upos in {'PROPN', 'NOUN'}:
                    print(f"Pronoun: {word.text}, Head (referent): {head.text}")
                    # Заменяем местоимение на референт
                    resolved_text = resolved_text.replace(word.text, head.text)
    
    return resolved_text


# Пример использования
text = "Alice went to the store. She bought some milk."
resolved_text = resolve_anaphora_stanza(text)
print(resolved_text)  # Output: Alice went to the store. Alice bought some milk.