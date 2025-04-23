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
    last_referents = {'he': None, 'she': None, 'it': None, 'they': None}
    
    # Перебираем все предложения в документе
    for sent in doc.sentences:
        for word in sent.words:
            # Ищем местоимения (например, 'it', 'he', 'she') и пытаемся найти их референты
            if word.upos in {'PROPN', 'NOUN'}:
                # Обновляем референты по роду
                if word.feats:
                    feats = word.feats
                    if 'Gender=Masc' in feats:
                        last_referents['he'] = word
                    elif 'Gender=Fem' in feats:
                        last_referents['she'] = word
                    else:
                        last_referents['it'] = word  # по умолчанию
                else:
                    last_referents['it'] = word  # fallback
            elif word.upos == 'PRON':
                pronoun = word.text.lower()
                referent = last_referents.get(pronoun)
                if referent:
                    print(f"Replacing pronoun '{word.text}' with '{referent.text}'")
                    resolved_text = resolved_text.replace(word.text, referent.text)
    
    return resolved_text


if __name__ == "__main__":
    # Пример использования
    text = "Alice went to the store. She bought some milk."
    resolved_text = resolve_anaphora_stanza(text)
    print(resolved_text)  # Output: Alice went to the store. Alice bought some milk.