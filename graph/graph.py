from typing import Dict, Set, List, Optional
from collections import defaultdict

from graph.edge import Edge
from graph.vertex import Vertex

import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    """
    Represents a graph structure with vertices and edges.

    Attributes:
        vertices: Dictionary mapping concept names to Vertex objects
        edges: List of Edge objects
        vertex_edges: Dictionary mapping vertex concepts to sets of edge indices
    """

    def __init__(self) -> None:
        self.vertices: Dict[str, Vertex] = {}
        self.edges: List[Edge] = []
        self.vertex_edges: Dict[str, Set[int]] = defaultdict(set)

    def add_vertex(
        self, concept: str, words_of_concept: Optional[List[str]] = None
    ) -> None:
        """
        Add a vertex to the graph.

        Args:
            concept: The main concept for the vertex
            words_of_concept: Associated words (defaults to [concept] if None)

        Raises:
            ValueError: If the vertex already exists
        """
        if concept in self.vertices:
            raise ValueError(f"Vertex with concept '{concept}' already exists")

        if words_of_concept is None:
            words_of_concept = [concept]

        self.vertices[concept] = Vertex(concept, words_of_concept)

    def add_edge(
        self,
        agent_1: str,
        agent_2: str,
        meaning: str,
        edge_type: int = 1,
        parent_subgraph: int = 1,
    ) -> None:
        """
        Add an edge between two vertices.

        Args:
            word_1: Source vertex concept (if applicable)
            word_2: Target vertex concept (if applicable)
            meaning: Relationship meaning
            edge_type: Type of the edge
            parent_subgraph: ID of the parent subgraph or source

        Raises:
            ValueError: If either vertex doesn't exist
        """
        # Validate vertices exist
        if agent_1 not in self.vertices:
            raise ValueError(f"Agent 1 vertex '{agent_1}' does not exist")
        if agent_2 not in self.vertices:
            raise ValueError(f"Agent 2 vertex '{agent_2}' does not exist")

        # Create and store the edge
        edge_index = len(self.edges)
        new_edge = Edge(agent_1, agent_2, meaning, edge_type, parent_subgraph)
        self.edges.append(new_edge)

        # Update vertex adjacency information
        self.vertex_edges[agent_1].add(edge_index)
        self.vertex_edges[agent_2].add(edge_index)

        # Update vertex objects
        self.vertices[agent_1].adjacent_edges.add(edge_index)
        self.vertices[agent_2].adjacent_edges.add(edge_index)

    def get_vertex_edges(self, concept: str) -> List[Edge]:
        """
        Get all edges connected to a vertex.

        Args:
            concept: The vertex concept to get edges for

        Returns:
            List of Edge objects connected to the vertex

        Raises:
            ValueError: If the vertex doesn't exist
        """
        if not isinstance(concept, str):
            raise ValueError("Concept must be a string")

        if concept not in self.vertices:
            raise ValueError(f"Vertex '{concept}' does not exist")

        return [self.edges[i] for i in self.vertex_edges[concept]]

    def get_edges(self) -> List[Edge]:
        """
        Get all edges connected to a vertex.

        Returns:
            List of all Edge objects in graph
        """
        return self.edges

    def __str__(self) -> str:
        return f"Graph(vertices={len(self.vertices)}, edges={len(self.edges)})"

    def __repr__(self) -> str:
        return (
            f"Graph(\n\tvertices={list(self.vertices.values())},\n"
            f"\tedges={self.edges}\n)"
        )


def visualize_graph(graph: Graph) -> None:
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes (vertices)
    for concept, _ in graph.vertices.items():
        G.add_node(concept)

    # Add edges with their meanings
    for edge in graph.edges:
        G.add_edge(edge.agent_1, edge.agent_2, meaning=edge.meaning)

    # Create the layout
    pos = nx.spring_layout(G)

    # Draw the graph
    plt.figure(figsize=(12, 8))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=2000, alpha=0.7)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, "meaning")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

    plt.title("Concept Tree")
    plt.axis("off")
    plt.tight_layout()
    plt.show()