import os
import sys
import numpy as np
from typing import List, Dict, Union
from collections import Counter


from directed_graph.graph import Graph
from directed_graph.edge import Edge
from directed_graph.vertex import Vertex



class GraphMetrics:
    """Calculates various metrics for a given Graph object."""

    def __init__(self, graph: Graph):
        self.graph = graph

    def degree(self, concept: str) -> int:
        """Calculates the degree of a vertex."""
        if concept not in self.graph.vertices:
            raise ValueError(f"Vertex '{concept}' does not exist.")
        return len(self.graph.get_vertex_edges(concept))

    def average_degree(self) -> float:
        """Calculates the average degree of the graph."""
        if not self.graph.vertices:
            return 0.0  # Handle empty graph case
        total_degree = sum(self.degree(concept) for concept in self.graph.vertices)
        return total_degree / len(self.graph.vertices)

    def degree_distribution(self) -> Dict[int, float]:
        """Calculates the degree distribution of the graph."""
        degrees = [self.degree(concept) for concept in self.graph.vertices]
        degree_counts = Counter(degrees)
        total_nodes = len(self.graph.vertices)
        return {degree: count / total_nodes for degree, count in degree_counts.items()}

    def shortest_path_length(
        self, start_concept: str, end_concept: str
    ) -> Union[int, float]:
        """Calculates the shortest path length between two vertices."""
        if (
            start_concept not in self.graph.vertices
            or end_concept not in self.graph.vertices
        ):
            raise ValueError("Start or end vertex does not exist.")

        if start_concept == end_concept:
            return 0  # Distance to itself is 0

        # BFS to find shortest path
        queue = [(start_concept, 0)]  # (node, distance)
        visited = {start_concept}

        while queue:
            current_node, distance = queue.pop(0)
            if current_node == end_concept:
                return distance

            for edge in self.graph.get_vertex_edges(current_node):
                neighbor = (
                    edge.agent_2 if edge.agent_1 == current_node else edge.agent_1
                )
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))
                    visited.add(neighbor)

        return float("inf")  # No path exists

    def average_shortest_path_length(self) -> Union[float, None]:
        """Calculates the average shortest path length for the graph."""
        if not self.graph.vertices:
            return None  # Handle empty graph case

        total_path_length = 0
        num_pairs = 0
        vertices = list(self.graph.vertices.keys())

        for i in range(len(vertices)):
            for j in range(
                i + 1, len(vertices)
            ):  # Avoid double counting and self-loops
                path_length = self.shortest_path_length(vertices[i], vertices[j])
                if path_length != float("inf"):
                    total_path_length += path_length
                    num_pairs += 1

        return (total_path_length / num_pairs) if num_pairs > 0 else float("inf")

    def diameter(self) -> Union[int, float]:
        """Calculates the diameter of the graph."""
        if not self.graph.vertices:
            return 0.0  # Handle empty graph case

        max_path_length = 0
        vertices = list(self.graph.vertices.keys())

        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                path_length = self.shortest_path_length(vertices[i], vertices[j])
                max_path_length = max(max_path_length, path_length)

        return max_path_length if max_path_length != float("inf") else float("inf")

    def clustering_coefficient(self, concept: str) -> Union[float, int]:
        """Calculates the clustering coefficient of a vertex."""
        if concept not in self.graph.vertices:
            raise ValueError(f"Vertex '{concept}' does not exist.")

        neighbors = set()
        for edge in self.graph.get_vertex_edges(concept):
            neighbors.add(edge.agent_2 if edge.agent_1 == concept else edge.agent_1)

        k = len(neighbors)
        if k < 2:
            return 0  # Clustering coefficient is 0 for vertices with degree < 2

        possible_connections = k * (k - 1) // 2
        actual_connections = 0

        neighbors_list = list(neighbors)  # create a list
        for i in range(k):
            for j in range(i + 1, k):
                try:
                    self.shortest_path_length(
                        neighbors_list[i], neighbors_list[j]
                    )  # Check for path
                    actual_connections += 1
                except ValueError:
                    pass

        return actual_connections / possible_connections

    def average_clustering_coefficient(self) -> float:
        """Calculates the average clustering coefficient of the graph."""
        if not self.graph.vertices:
            return 0.0  # Handle empty graph case

        total_clustering_coefficient = sum(
            self.clustering_coefficient(concept) for concept in self.graph.vertices
        )
        return total_clustering_coefficient / len(self.graph.vertices)

    def assortativity(self) -> float:
        """Calculates the assortativity coefficient of the graph."""
        if not self.graph.edges:
            return 0.0

        sum_jeke = 0.0
        sum_j_plus_k = 0.0
        sum_j2_plus_k2 = 0.0

        E = len(self.graph.edges)

        for edge in self.graph.edges:
            agent_1 = edge.agent_1
            agent_2 = edge.agent_2

            j_e = self.degree(agent_1)
            k_e = self.degree(agent_2)

            sum_jeke += j_e * k_e
            sum_j_plus_k += j_e + k_e
            sum_j2_plus_k2 += j_e**2 + k_e**2

        r = (sum_jeke / E - (sum_j_plus_k / (2 * E)) ** 2) / (
            (sum_j2_plus_k2 / (2 * E)) - (sum_j_plus_k / (2 * E)) ** 2
        )
        return r