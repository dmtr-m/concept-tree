from typing import Dict, Set, List, Optional
from collections import defaultdict

from graph.edge import Edge
from graph.vertex import Vertex
from graph.union_edge import UnionEdge

from typing import List, Optional, Set, Dict
from collections import defaultdict


class Graph:
    """
    Represents a graph structure with vertices and edges, supporting union edges.
    """

    def __init__(self) -> None:
        self.vertices: Dict[str, Vertex] = {}
        self.edges: List[Edge] = []
        self.vertex_edges: Dict[str, Set[int]] = defaultdict(set)
        self.union_edges: List[UnionEdge] = []

    def add_union_edge(self, agent_group_1: List[Edge], agent_2: str,
                       meaning: str, parent_subgraph: int) -> None:
        """
        Add edges representing a definition with multiple parts.

        Args:
            edge_group_1: List of tuples representing the first part of the higher dimensional graph
            agent_2: Agent 2 of the higher dimensional graph
            meaning: Relationship meaning
            parent_subgraph: ID of the parent subgraph/source
        """
        edge_indices = set()

        # Add individual edges from each part of the higher dimensional graph
        def add_individual_edges(group: List[Edge]) -> None:
            for edge in group:
                edge_index = len(self.edges)
                if edge.agent_1 not in self.vertices:
                    self.add_vertex(edge.agent_1)
                if edge.agent_2 not in self.vertices:
                    self.add_vertex(edge.agent_2)

            self.add_edge(edge.agent_1, edge.agent_2, edge.meaning,
                          edge.edge_type, parent_subgraph)
            edge_indices.add(edge_index)

        add_individual_edges(agent_group_1)

        if agent_2 not in self.vertices:
            self.add_vertex(agent_2)

        added_edges = set()
        for edge in agent_group_1:
            if edge.agent_1 not in added_edges:
                edge_index = len(self.edges)
                self.add_edge(edge.agent_1, agent_2, "part_" + meaning,
                              edge.edge_type, parent_subgraph)
                edge_indices.add(edge_index)
            if edge.agent_2 not in added_edges:
                edge_index = len(self.edges)
                self.add_edge(edge.agent_2, agent_2, "part_" + meaning,
                              edge.edge_type, parent_subgraph)
                edge_indices.add(edge_index)

        # Create the union edge grouping
        union_edge = UnionEdge(
            edge_ids=edge_indices,
            meaning=meaning,
            parent_subgraph=parent_subgraph,
        )
        self.union_edges.append(union_edge)

    def get_union_edges(self) -> List[Dict]:
        """
        Get all union edges with their component edges.

        Returns:
            List of dictionaries containing union edge information
        """
        result = []
        for union_edge in self.union_edges:
            edges = [self.edges[i] for i in union_edge.edge_ids]
            result.append({
                "type": union_edge.union_type,
                "parent_subgraph": union_edge.parent_subgraph,
                "edges": edges
            })
        return result

    def add_vertex(self, concept: str, words_of_concept: Optional[List[str]] = None) -> None:
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

    def add_edge(self, agent_1: str, agent_2: str, meaning: str,
                 edge_type: int, parent_subgraph: int) -> None:
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
        if concept not in self.vertices:
            raise ValueError(f"Vertex '{concept}' does not exist")

        return [self.edges[i] for i in self.vertex_edges[concept]]

    def __str__(self) -> str:
        return f"Graph(vertices={len(self.vertices)}, edges={len(self.edges)})"

    def __repr__(self) -> str:
        return (f"Graph(\n\tvertices={list(self.vertices.values())},\n"
                f"\tedges={self.edges}\n)")
