import os

os.environ["CORENLP_HOME"] = "corenlp_graph_maker/stanford-corenlp-4.5.8"

from stanza.server import CoreNLPClient


class OpenIEProcessor:
    def __init__(self, endpoint="http://localhost:9001", timeout=30000):
        """
        Инициализация процессора OpenIE.
        :param annotators: Список аннотаторов CoreNLP.
        :param endpoint: Адрес сервера CoreNLP.
        :param timeout: Таймаут для клиента.
        """

        self.properties = {
            "annotators": "tokenize,ssplit,pos,lemma,ner,openie",
            "openie.triple.strict": "true",
            "openie.splitConjuncts": "true",
        }

        self.endpoint = endpoint
        self.timeout = timeout

    def extract_triples(self, text, lemmatize=False):
        """
        Извлечение триплетов из текста с возможностью лемматизации в контексте предложения.
        :param text: Исходный текст.
        :param lemmatize: Флаг для включения лемматизации (по умолчанию False).
        :return: Список триплетов.
        """
        triples = []

        # Используем контекстный менеджер для управления клиентом CoreNLP
        with CoreNLPClient(
            properties=self.properties,
            endpoint=self.endpoint,
            timeout=self.timeout,
            be_quiet=True,
        ) as client:
            # Аннотация текста
            document = client.annotate(text, output_format="json")

            # Извлечение триплетов
            for sentence in document["sentences"]:
                # Лемматизация всех токенов в предложении
                if lemmatize:
                    word_to_lemma = {
                        token["word"]: token["lemma"] for token in sentence["tokens"]
                    }

                for triple in sentence.get("openie", []):
                    subject = triple["subject"]
                    relation = triple["relation"]
                    obj = triple["object"]

                    # Лемматизация элементов триплета с учетом контекста
                    if lemmatize:
                        subject_lemma = " ".join(
                            [word_to_lemma.get(word, word) for word in subject.split()]
                        )
                        relation_lemma = " ".join(
                            [word_to_lemma.get(word, word) for word in relation.split()]
                        )
                        object_lemma = " ".join(
                            [word_to_lemma.get(word, word) for word in obj.split()]
                        )
                    else:
                        subject_lemma = subject
                        relation_lemma = relation
                        object_lemma = obj

                    # Добавление триплета
                    triples.append(
                        {
                            "subject": subject_lemma,
                            "relation": relation_lemma,
                            "object": object_lemma,
                        }
                    )

        return triples


# Пример использования
if __name__ == "__main__":
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

    # Создание экземпляра класса
    processor = OpenIEProcessor()

    # Извлечение триплетов
    triples = processor.extract_triples(text, lemmatize=True)

    # Вывод результатов
    print("Извлеченные триплеты:")
    for triple in triples:
        print(triple)
