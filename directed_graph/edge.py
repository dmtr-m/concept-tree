from dataclasses import dataclass, field

from directed_graph.embedding_manager_fasttext import get_embedding

import numpy as np


@dataclass
class Edge:
    """
    Represents an edge in the graph connecting two vertices.

    Attributes:
        agent_1: Source vertex concept (if applicable)
        agent_2: Target vertex concept (if applicable)
        meaning: The relationship meaning
        edge_type: Type of the edge
        parent_subgraph: ID of the parent text or source
        label_pos: label part of speech
        embedding: Embedding of the meaning
    """

    agent_1: str
    agent_2: str
    label: str
    edge_type: int
    parent_subgraph: int
    label_pos: str
    embedding: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.embedding = get_embedding(self.label.split())

    def __repr__(self) -> str:
        return f"Edge({self.agent_1} <--[{self.label}]--> {self.agent_2})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, value):
        return (
            self.agent_1 == value.agent_1
            and self.agent_2 == value.agent_2
            and self.label == value.meaning
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.agent_1,
                self.agent_2,
                self.label,
                self.edge_type,
                self.parent_subgraph,
            )
        )
