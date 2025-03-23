import os

# requires corenlpserver https://nlp.stanford.edu/software/stanford-corenlp-4.5.8.zip
os.environ["CORENLP_HOME"] = "corenlp_triples_maker/stanford-corenlp-4.5.8"

import os
import pandas as pd
from stanza.server import CoreNLPClient


class OpenIETriplesMaker:
    def __init__(self, endpoint="http://localhost:9000", max_length=3):
        self.endpoint = endpoint
        self.max_length = max_length
        self.max_threads = os.cpu_count()  # Получаем количество ядер CPU
        print(f"Cpu count: {self.max_threads}")

        # Настройка properties
        self.properties = {
            "annotators": "tokenize,ssplit,pos,lemma,ner,openie",
            "openie.triple.strict": "true",
            "openie.splitConjuncts": "true",
        }

    def extract_triples(self, text):
        """
        Извлечение и фильтрация триплетов из текста.
        :param text: Исходный текст.
        :return: Список отфильтрованных триплетов с POS для relation.
        """
        triples = []

        # Используем контекстный менеджер для управления клиентом CoreNLP
        with CoreNLPClient(
            properties=self.properties,
            threads=self.max_threads,
            endpoint=self.endpoint,
            timeout=30000,
            be_quiet=True,
        ) as client:
            # Аннотация текста
            document = client.annotate(text)

            # Извлечение и фильтрация триплетов
            for sentence in document.sentence:
                # Создаем карты для лемматизации и POS
                word_to_lemma = {
                    token.word.lower(): token.lemma.lower() for token in sentence.token
                }
                word_to_pos = {
                    token.word.lower(): token.pos for token in sentence.token
                }

                for triple in sentence.openieTriple:
                    # Лемматизация элементов триплета
                    subject_lemma = " ".join(
                        [
                            word_to_lemma.get(word.lower(), word.lower())
                            for word in triple.subject.split()
                        ]
                    )
                    relation_lemma = " ".join(
                        [
                            word_to_lemma.get(word.lower(), word.lower())
                            for word in triple.relation.split()
                        ]
                    )
                    object_lemma = " ".join(
                        [
                            word_to_lemma.get(word.lower(), word.lower())
                            for word in triple.object.split()
                        ]
                    )

                    # Определение POS для relation с учетом контекста
                    relation_pos = []
                    for word in triple.relation.split():
                        if word.lower() in word_to_pos:
                            relation_pos.append(word_to_pos[word.lower()])
                        else:
                            relation_pos.append("UNKNOWN")
                    relation_pos = " ".join(relation_pos)

                    # Проверка длины элементов перед добавлением
                    if (
                        len(subject_lemma.split()) <= self.max_length
                        and len(relation_lemma.split()) <= self.max_length
                        and len(object_lemma.split()) <= self.max_length
                    ):
                        triples.append(
                            {
                                "subject": subject_lemma,
                                "relation": relation_lemma,
                                "relation_pos": relation_pos,  # Добавляем POS для relation
                                "object": object_lemma,
                            }
                        )

        return triples

    def save_triples_to_csv(self, triples, output_file):
        """
        Сохранение триплетов в формате CSV с использованием pandas.
        :param triples: Список триплетов.
        :param output_file: Путь к выходному файлу CSV.
        """
        # Создание DataFrame из списка триплетов
        df = pd.DataFrame(
            triples, columns=["subject", "relation", "relation_pos", "object"]
        )

        # Сохранение в CSV
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"Триплеты сохранены в файл: {output_file}")


# Пример использования
if __name__ == "__main__":
    processor = OpenIETriplesMaker(max_length=3)

    # Исходный текст
    text = """
    Biological systems reach organizational complexity that far exceeds the complexity of any
    known inanimate objects. Biological entities undoubtedly obey the laws of quantum physics and
    statistical mechanics. However, is modern physics sufficient to adequately describe, model and
    explain the evolution of biological complexity? Detailed parallels have been drawn between
    statistical thermodynamics and the population-genetic theory of biological evolution. Based on
    these parallels, we outline new perspectives on biological innovation and major transitions in
    evolution, and introduce a biological equivalent of thermodynamic potential that reflects the
    innovation propensity of an evolving population. Deep analogies have been suggested to also
    exist between the properties of biological entities and processes, and those of frustrated states in
    physics, such as glasses. Such systems are characterized by frustration whereby local state with
    minimal free energy conflict with the global minimum, resulting in “emergent phenomena”. We
    extend such analogies by examining frustration-type phenomena, such as conflicts between
    different levels of selection, in biological evolution. These frustration effects appear to drive the
    evolution of biological complexity. We further address evolution in multidimensional fitness
    landscapes from the point of view of percolation theory and suggest that percolation at level
    above the critical threshold dictates the tree-like evolution of complex organisms. Taken
    together, these multiple connections between fundamental processes in physics and biology
    imply that construction of a meaningful physical theory of biological evolution might not be a
    futile effort. However, it is unrealistic to expect that such a theory can be created in one scoop;
    if it ever comes to being, this can only happen through integration of multiple physical models of
    evolutionary processes. Furthermore, the existing framework of theoretical physics is unlikely to
    suffice for adequate modeling of the biological level of complexity, and new developments
    within physics itself are likely to be required.
    """

    # Извлечение и фильтрация триплетов
    triples = processor.extract_triples(text)

    # Вывод результатов
    print("Отфильтрованные триплеты:")
    for triple in triples:
        print(triple)

    # Сохранение триплетов в CSV
    output_file = "filtered_triples.csv"
    processor.extract_triples_to_csv(triples, output_file)
