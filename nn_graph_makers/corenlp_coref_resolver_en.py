import os

# requires corenlpserver https://nlp.stanford.edu/software/stanford-corenlp-4.5.8.zip
os.environ["CORENLP_HOME"] = "corenlp_triples_maker/stanford-corenlp-4.5.8"

from stanza.server import CoreNLPClient


class CoreferenceResolver:
    def __init__(self, endpoint="http://localhost:9000"):
        self.endpoint = endpoint

    def resolve_coreferences(self, text, algorithm="neural"):
        """
        Разрешение анафорических ссылок в тексте.
        :param text: Исходный текст.
        :param algorithm: Алоритм разрешения анафоры: neural, statistical
        :return: Текст с разрешенными анафорическими ссылками.
        """
        self.max_threads = os.cpu_count()  # Получаем количество ядер CPU

        # Настройка properties
        self.properties = {
            "annotators": "tokenize,ssplit,pos,lemma,ner,parse,coref",
            "coref.algorithm": algorithm,
            "threads": f"{self.max_threads}",  # Используем все доступные потоки
        }

        with CoreNLPClient(
            properties=self.properties,
            endpoint=self.endpoint,
            timeout=60000,  # Увеличиваем таймаут
            be_quiet=True,
        ) as client:
            # Аннотируем текст
            document = client.annotate(text)

            # Создаем список для хранения замен
            replacements = []

            # Проходим по всем цепочкам кореференции
            for chain in document.corefChain:
                # Получаем основное упоминание (representative mention)
                representative_mention = chain.mention[chain.representative]
                representative_text = " ".join(
                    token.word
                    for token in document.sentence[
                        representative_mention.sentenceIndex
                    ].token[
                        representative_mention.beginIndex : representative_mention.endIndex
                    ]
                )

                # Проходим по всем упоминаниям в цепочке
                for mention in chain.mention:
                    if mention.mentionID != representative_mention.mentionID:
                        # Заменяем анафорическое выражение на основное упоминание
                        mention_text = " ".join(
                            token.word
                            for token in document.sentence[mention.sentenceIndex].token[
                                mention.beginIndex : mention.endIndex
                            ]
                        )
                        replacements.append(
                            {
                                "sentence_index": mention.sentenceIndex,
                                "start": document.sentence[mention.sentenceIndex]
                                .token[mention.beginIndex]
                                .beginChar,
                                "end": document.sentence[mention.sentenceIndex]
                                .token[mention.endIndex - 1]
                                .endChar,
                                "replacement": representative_text,
                            }
                        )

            # Применяем замены в тексте (с конца к началу, чтобы избежать смещения индексов)
            resolved_text = text
            for replacement in sorted(
                replacements, key=lambda x: x["start"], reverse=True
            ):
                resolved_text = (
                    resolved_text[: replacement["start"]]
                    + replacement["replacement"]
                    + resolved_text[replacement["end"] :]
                )

            return resolved_text


# Пример использования
if __name__ == "__main__":
    resolver = CoreferenceResolver()

    text = """
        John decided to go to the park.

        He wanted to relax after a long day at work.

        The park was quiet, and John found a bench to sit on.

        He noticed a dog running around nearby.

        The dog seemed friendly, and it approached him.

        John petted the dog, and it wagged its tail happily.

        After a while, the dog ran off to play with another dog.

        John watched them for a few minutes before deciding to leave.

        On his way home, he stopped by a café.

        The café was cozy, and he ordered a cup of coffee.

        While drinking his coffee, John opened a book he had brought with him.

        The book was about history, and it was quite interesting.

        He read for about an hour before realizing it was getting late.

        John paid for his coffee and left the café.

        As he walked home, he thought about the book.

        It had given him a lot to think about.

        When John arrived home, he called his friend Sarah.

        She was excited to hear about his day.

        Sarah told him about her own day, and they chatted for a while.

        After the call, John felt happy and went to bed.
    """

    resolved_text = resolver.resolve_coreferences(text, "statistical")
    print("Текст с разрешенной анафорой:")
    print(resolved_text)
