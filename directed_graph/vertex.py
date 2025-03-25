from dataclasses import dataclass, field
from typing import List, Set

from directed_graph.embedding_manager import get_embedding

import numpy as np


@dataclass
class Vertex:
    """
    Represents a vertex in the graph with a concept and its associated words.

    Attributes:
        concept: The concept this vertex represents
        words_of_concept: List of words of this concept
        embedding: The embedding of the concept
        adjacent_edges: Set of adjacent edge IDs
    """

    concept: str
    words_of_concept: List[str]
    vertex_type: int
    embedding: np.ndarray = field(init=False)
    incoming_edges: Set[int] = field(default_factory=set)
    outgoing_edges: Set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.embedding = get_embedding(self.words_of_concept)

    def __repr__(self) -> str:
        return f"Vertex(concept='{self.concept}', words={self.words_of_concept})"

    def __hash__(self):
        return hash(self.concept)
