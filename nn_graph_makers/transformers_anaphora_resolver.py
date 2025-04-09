from transformers import pipeline

# Создание пайплайна для разрешения кореференций
coref_pipeline = pipeline("coreference-resolution", model="SpanBERT/spanbert-base-cased")

def resolve_anaphora(text):
    result = coref_pipeline(text)
    resolved_text = text
    for cluster in result["clusters"]:
        main_mention = " ".join(result["document"][cluster[0][0]:cluster[0][1] + 1])
        for mention in cluster[1:]:
            mention_text = " ".join(result["document"][mention[0]:mention[1] + 1])
            resolved_text = resolved_text.replace(mention_text, main_mention)
    return resolved_text