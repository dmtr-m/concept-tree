import spacy
from typing import List, Dict, Tuple, Set
from directed_graph.graph import Graph, visualize_graph
from collections import defaultdict


class SentenceAnalyzer:
    def __init__(self):
        # Load Russian language model
        self.nlp = spacy.load("en_core_web_trf")
        self.graph = Graph()

    def extract_noun_phrases(self, doc) -> List[Tuple[spacy.tokens.Span, List[str]]]:
        """
        Extract noun phrases and their component words from the document.

        Returns:
            List of tuples containing (noun_phrase_span, list_of_words)
        """
        noun_phrases = []
        for chunk in doc.noun_chunks:
            # Get the root of the noun phrase
            root = chunk.root

            # Collect all words in the noun phrase
            words = [token.text.lower() for token in chunk]

            # Create the full noun phrase text
            full_phrase = " ".join(words)

            noun_phrases.append((chunk, words))

        return noun_phrases

    def find_verb_connections(self, doc, noun_phrases: List[Tuple[spacy.tokens.Span, List[str]]]) -> List[Tuple[str, str, str]]:
        """
        Find verbal connections between noun phrases.

        Returns:
            List of tuples containing (source_np, target_np, verb)
        """
        connections = []
        np_spans = [np[0] for np in noun_phrases]
        np_dict = {np.text.lower(): words for np, words in noun_phrases}

        for token in doc:
            if token.pos_ == "VERB":
                # Find subject and object connected to this verb
                subj = None
                obj = None
                prep_phrase = None

                # Look for subject and object in dependencies
                for child in token.children:
                    if child.dep_ == "nsubj":
                        # Find the corresponding noun phrase
                        for np in np_spans:
                            if child in np:
                                subj = np.text.lower()
                                break
                    elif child.dep_ in ["dobj", "iobj"]:
                        for np in np_spans:
                            if child in np:
                                obj = np.text.lower()
                                break
                    elif child.dep_ == "prep":
                        # Look for noun phrases in prepositional phrase
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                for np in np_spans:
                                    if grandchild in np:
                                        prep_phrase = (np.text.lower(), f"{token.text} {child.text}")
                                        break

                # Add connections if found
                if subj and obj:
                    connections.append((subj, obj, token.text))
                if subj and prep_phrase:
                    connections.append((subj, prep_phrase[0], prep_phrase[1]))

        return connections

    def analyze_sentence(self, text: str) -> None:
        """
        Analyze a sentence and build the corresponding graph.
        """
        # Process the text
        doc = self.nlp(text)

        # Extract noun phrases
        noun_phrases = self.extract_noun_phrases(doc)

        # Add vertices for each noun phrase
        for _, words in noun_phrases:
            concept = " ".join(words)
            if concept not in self.graph.vertices:
                self.graph.add_vertex(concept, words)

        # Find and add connections
        connections = self.find_verb_connections(doc, noun_phrases)

        # Add edges to the graph
        for source, target, relation in connections:
            try:
                self.graph.add_edge(source, target, relation,
                                    1, len(self.graph.edges))
            except ValueError as e:
                print(f"Warning: Could not add edge: {e}")

    def get_graph(self) -> Graph:
        """Return the constructed graph."""
        return self.graph

# Example usage


def main():
    analyzer = SentenceAnalyzer()

    # Example sentences
    sentences = [
        "They have attracted a wide adult audience as well as younger readers and are widely considered cornerstones of modern literature, though the books have received mixed reviews from critics and literary scholars."
    ]

    # Process each sentence
    for sentence in sentences:
        analyzer.analyze_sentence(sentence)

    # Get and print the resulting graph
    graph = analyzer.get_graph()
    print("Graph structure:")
    print(graph)
    print("\nDetailed representation:")
    print(repr(graph))

    # Print edges for each concept
    print("\nEdges:")
    for concept in graph.vertices.keys():
        print(f"\nConcept: {concept}")
        print(graph.get_vertex_edges(concept))

    visualize_graph(graph)


if __name__ == "__main__":
    main()