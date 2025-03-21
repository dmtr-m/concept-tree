from directed_graph.graph import Graph

import spacy
from spacy import displacy


class GraphMaker:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("coreferee")

    def create_dependency_graph(self, sentence):
        doc = self.nlp(self.resolve_coreferences(sentence))

        G = Graph()
        verb_noun_dict = {}

        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "VERB":
                    verb = (token.text, token.i)
                    if verb not in verb_noun_dict:
                        verb_noun_dict[verb] = set()

                    for child in token.children:
                        if child.pos_ == "NOUN" and child.text.isalpha():
                            noun = (child.text, child.i)
                            verb_noun_dict[verb].add(noun)

        for verb, nouns in verb_noun_dict.items():
            for noun1 in nouns:
                if not G.contains_vertex(noun1[0]):
                    G.add_vertex(noun1[0])
                for noun2 in nouns:
                    if noun1 != noun2:
                        if not G.contains_vertex(noun2[0]):
                            G.add_vertex(noun2[0])
                        if noun1[1] < noun2[1]:
                            G.add_edge(noun1[0], noun2[0], verb[0])
                        else:
                            G.add_edge(noun2[0], noun1[0], verb[0])

        return G

    def create_dependency_graph_bi(self, sentence):
        doc = self.nlp(self.resolve_coreferences(sentence))

        G = Graph()
        verb_bigram_dict = {}

        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "VERB":
                    verb = (token.text, token.i)
                    if verb not in verb_bigram_dict:
                        verb_bigram_dict[verb] = set()

                    for child in token.children:
                        if (
                            child.pos_ == "NOUN"
                            and child.text.isalpha()
                            and len(child.text) > 1
                        ):
                            noun = (child.text, child.i)
                            for grandchild in child.children:
                                if (
                                    grandchild.text.isalpha()
                                    and grandchild.pos_
                                    not in ["VERB", "CONJ", "DET", "ADP"]
                                    and len(grandchild.text) > 1
                                ):
                                    dependent_word = (grandchild.text, grandchild.i)

                                    bigram = tuple(
                                        sorted(
                                            [noun, dependent_word], key=lambda x: x[1]
                                        )
                                    )
                                    verb_bigram_dict[verb].add(bigram)

        for verb, bigrams in verb_bigram_dict.items():
            for bigram1 in bigrams:
                if not G.contains_vertex(" ".join(bigram1)):
                    G.add_vertex(" ".join(bigram1))
                for bigram2 in bigrams:
                    if (
                        bigram1 != bigram2
                        and not bigram1[0] in bigram2
                        and not bigram1[1] in bigram2
                        and not bigram2[0] in bigram1
                        and not bigram2[1] in bigram1
                    ):
                        if not G.contains_vertex(" ".join(bigram2)):
                            G.add_vertex(" ".join(bigram2))
                        if bigram1[0][1] < bigram2[0][1]:
                            G.add_edge(" ".join(bigram1), " ".join(bigram2), verb[0])
                        else:
                            G.add_edge(" ".join(bigram2), " ".join(bigram1), verb[0])

        return G

    def create_dependency_graph_tri(self, sentence):
        doc = self.nlp(self.resolve(sentence))

        G = Graph()
        verb_trigram_dict = {}
        embeddings = {}

        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "VERB":
                    verb = (token.text, token.i)
                    if verb not in verb_trigram_dict:
                        verb_trigram_dict[verb] = set()

                    for child in token.children:
                        if (
                            child.pos_ == "NOUN"
                            and child.text.isalpha()
                            and len(child.text) > 1
                        ):
                            noun = (child.text, child.i)
                            for grandchild in child.children:
                                if (
                                    grandchild.text.isalpha()
                                    and grandchild.pos not in ["VERB", "DET"]
                                    and len(grandchild.text) > 1
                                ):
                                    dependent_word = (grandchild.text, grandchild.i)
                                    for grandgrandchild in grandchild.children:
                                        if (
                                            grandgrandchild.pos
                                            not in ["VERB", "DET", "ADP"]
                                            and grandgrandchild.text.isalpha()
                                            and len(grandgrandchild.text) > 1
                                        ):
                                            last_word = (
                                                grandgrandchild.text,
                                                grandgrandchild.i,
                                            )

                                            trigram = tuple(
                                                sorted(
                                                    [noun, dependent_word, last_word],
                                                    key=lambda x: x[1],
                                                )
                                            )
                                            verb_trigram_dict[verb].add(trigram)

        for verb, trigrams in verb_trigram_dict.items():
            for trigram1 in trigrams:
                if not G.contains_vertex(" ".join(trigram1)):
                    G.add_vertex(" ".join(trigram1))
                for trigram2 in trigrams:
                    if (
                        trigram1 != trigram2
                        and not trigram1[0] in trigram2
                        and not trigram1[1] in trigram2
                        and not trigram1[2] in trigram2
                        and not trigram2[0] in trigram1
                        and not trigram2[1] in trigram1
                        and not trigram2[2] in trigram1
                    ):
                        if not G.contains_vertex(trigram2):
                            G.add_vertex(trigram2)
                        if trigram1[0][1] < trigram2[0][1]:
                            G.add_edge(" ".join(trigram1), " ".join(trigram2), label=verb[0])
                        else:
                            G.add_edge(" ".join(trigram2), " ".join(trigram1), label=verb[0])

        return G, embeddings

    def resolve_coreferences(self, text):
        """
        Разрешение кореференций в тексте.

        Args:
            text (str): Исходный текст для анализа.

        Returns:
            str: Текст, где все местоимения заменены на их антецеденты.
        """
        # Обработка текста через spaCy
        doc = self.nlp(text)
        resolved_text = doc.text  # Копия исходного текста для модификации

        # Проходим по всем кластерам кореференций
        for chain in doc._.coref_chains:
            if len(chain) == 0:  # Пропускаем пустые кластеры
                continue

            # Основное упоминание (антецедент)
            try:
                antecedent = doc[chain[0][0] : chain[0][1]].text
            except IndexError:
                print(f"Error: Invalid indices in chain {chain}. Skipping...")
                continue

            # Заменяем каждое кореферентное упоминание на антецедент
            for mention in chain:
                if mention.is_coref:  # Проверяем, является ли упоминание кореферентным
                    mention_text = doc[mention[0] : mention[1]].text
                    resolved_text = resolved_text.replace(mention_text, antecedent, 1)

        return resolved_text
