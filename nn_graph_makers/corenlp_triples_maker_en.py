import os

# requires corenlpserver https://nlp.stanford.edu/software/stanford-corenlp-4.5.8.zip
os.environ["CORENLP_HOME"] = "nn_graph_makers/stanford-corenlp-4.5.8"

from directed_graph.graph import Graph, visualize_graph_with_equivalent_elements

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
    processor = OpenIETriplesMaker(max_length=3, endpoint="http://localhost:9090")

    # Исходный текст
    text = """
   Chemical reaction networks are naturally described by the mathematical theory of graphs in which chemical species are represented by vertices and physicochemical processes are represented by weighted and directed hyperedges which capture the direction and stoichiometric quantities of each process (Fig. 1 ) [[[cite]]] . 
   Graphs of this flavor admit a matrix representation: the incidence or stoichiometric matrix [[[formula]]] . Physically, these matrices satisfy the mass-action kinetic differential equation: [[[formula]]] (1) where [[[formula]]] is a vector of concentrations and [[[formula]]] is a vector of fluxes. Each [[[formula]]] corresponds to the net rate of change in concentrations of the chemical species involved in process [[[formula]]] . Each [[[formula]]] is a sum of directed fluxes: [[[formula]]] where directed fluxes are proportional to the probability of an encounter between reactants (or products) [[[cite]]] . 
   If the mathematical forms of these fluxes are known, the concentrations of chemicals in the network are readily obtained [[[cite]]] . 
   Without loss of generality, we will assume that all fluxes are reversible. 
   Specifically, processes proceed in the forward direction if [[[formula]]] , while for [[[formula]]] the process is reversed. 
   The graph representation of chemical reaction networks makes distance metrics on graphs attractive choices. 
   However, not all graph distance metrics are suitable for directed graphs with hyperedges; applications of these metrics typically ignore directionality or stoichiometry [[[cite]]] . 
   Other existing metrics opt for computational tractability, such as reducing the scope to comparisons of the presence/absence of metabolic processes [[[cite]]] or feature vectors of topological measures derived from a graph-theoretic approach [[[cite]]] . We avoid these information losses by leveraging advances in parallel computing and numerical linear algebra to make calculations of stoichiometric nullspaces tractable for large datasets [[[cite]]] . 
   At this point, one may ask why focus on nullspaces of stoichiometric matrices?

    """

    # Извлечение и фильтрация триплетов
    # triples = processor.extract_triples(text)

    # Вывод результатов
    # print("Отфильтрованные триплеты:")
    # for triple in triples:
        # print(triple)

    # Сохранение триплетов в CSV
    # output_file = "filtered_triples.csv"
    # processor.save_triples_to_csv(triples, output_file)

    graph = Graph.build_from_triples_csv("filtered_triples.csv")

    visualize_graph_with_equivalent_elements(graph)
