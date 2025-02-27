from dataclasses import dataclass, field

from graph.embedding_manager import EmbeddingManager

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
        embedding: Embedding of the meaning
    """
    agent_1: str
    agent_2: str
    meaning: str
    edge_type: int
    parent_subgraph: int
    embedding: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.embedding = EmbeddingManager.get_embedding(self.meaning)

    def __repr__(self) -> str:
        return f"Edge({self.agent_1} <--[{self.meaning}]--> {self.agent_2})"
    
    def __eq__(self, value):
        return self.agent_1 == value.agent_1 and self.agent_2 == value.agent_2 and self.meaning == value.meaning

    def __hash__(self) -> int:
        return hash((self.agent_1, self.agent_2, self.meaning, self.edge_type, self.parent_subgraph))
