from dataclasses import dataclass, field
from typing import Set

import numpy as np
from directed_graph.embedding_manager_fasttext import get_embedding


@dataclass
class UnionEdge:
    """
    Represents a group of related edges that form a logical union.

    Attributes:
        edge_ids: Set of indices of the edges that are part of this union
        label: The union meaning
        parent_subgraph: ID of the parent subgraph/source
        embedding: Embedding of the meaning
    """
    edge_ids: Set[int]
    label: str
    parent_subgraph: int
    embedding: np.ndarray = field(init=False)

    def __post_init__(self):
        self.embedding = get_embedding(self.label)
