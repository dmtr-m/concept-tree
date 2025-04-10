import networkx as nx

from graph.embedding_manager import get_embedding
from typing import List, Tuple

class Graph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_vertex(self, label: str):
        words_of_label = label.replace("[", "").replace("]", "").split()
        self.graph.add_node(
            label,
            words_of_label=words_of_label,
            embedding=get_embedding(words_of_label)
        )

    def add_edge(self, from_label: str, to_label: str, label: str):
        words_of_label = label.replace("[", "").replace("]", "").split()
        self.graph.add_edge(
            from_label,
            to_label,
            key=label,
            label=label,
            words_of_label=words_of_label,
            embedding=get_embedding(words_of_label)
        )

    def get_incoming_vertices(self, label: str) -> List[str]:
        return list(self.graph.predecessors(label))

    def get_outgoing_vertices(self, label: str) -> List[str]:
        return list(self.graph.successors(label))

    def get_all_edges(self) -> List[Tuple[str, str, str, dict]]:
        return [(u, v, k, d) for u, v, k, d in self.graph.edges(keys=True, data=True)]

    def get_all_vertices(self) -> List[Tuple[str, dict]]:
        return list(self.graph.nodes(data=True))

    def contains_vertex(self, label: str) -> bool:
        return label in self.graph.nodes
