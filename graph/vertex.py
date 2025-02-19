from dataclasses import dataclass, field
from typing import List, Set

from graph.embedding_manager import EmbeddingManager

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
    embedding: np.ndarray = field(init=False)
    adjacent_edges: Set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.embedding = EmbeddingManager.get_embedding(self.concept)

    def __repr__(self) -> str:
        return f"Vertex(concept='{self.concept}', words={self.words_of_concept})"

    def __hash__(self):
        return hash((self.concept, self.words_of_concept))
