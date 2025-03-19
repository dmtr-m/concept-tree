import os
import sys
import numpy as np
from typing import List, Dict, Union
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx

from directed_graph.graph import Graph
from directed_graph.edge import Edge
from directed_graph.vertex import Vertex

import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict


class GraphMetrics:
    """Calculates distributions and other metrics of a graph using NetworkX."""

    def __init__(self, graph):
        # Преобразуем пользовательский граф в объект NetworkX.Graph
        self.nx_graph = self._convert_to_networkx(graph)

    def _convert_to_networkx(self, graph) -> nx.Graph:
        """
        Converts a custom Graph object to a NetworkX Graph.
        Assumes the input graph is undirected for simplicity.
        """
        nx_graph = nx.Graph()

        # Add vertices (nodes)
        for vertex in graph.vertices:
            nx_graph.add_node(vertex)

        # Add edges
        for edge in graph.get_edges():
            nx_graph.add_edge(edge.agent_1, edge.agent_2)

        return nx_graph

    def degree_distribution(self) -> Dict[int, float]:
        """Calculates the degree distribution of the graph."""
        degrees = dict(self.nx_graph.degree())
        degree_counts = Counter(degrees.values())
        total_nodes = len(degrees)
        return {degree: count / total_nodes for degree, count in degree_counts.items()}

    def shortest_path_length_distribution(self) -> Dict[int, float]:
        """Calculates the distribution of shortest path lengths."""
        all_pairs_shortest_paths = dict(
            nx.all_pairs_shortest_path_length(self.nx_graph)
        )
        path_lengths = []
        for source, targets in all_pairs_shortest_paths.items():
            for target, length in targets.items():
                if source != target:  # Exclude self-loops
                    path_lengths.append(length)
        path_counts = Counter(path_lengths)
        total_pairs = len(path_lengths)
        return {length: count / total_pairs for length, count in path_counts.items()}

    def clustering_coefficient_distribution(self) -> Dict[float, float]:
        """Calculates the distribution of clustering coefficients."""
        clustering_coeffs = nx.clustering(self.nx_graph)
        coeff_counts = Counter(clustering_coeffs.values())
        total_nodes = len(clustering_coeffs)
        return {coeff: count / total_nodes for coeff, count in coeff_counts.items()}

    def connected_components(self) -> Dict[int, int]:
        """Calculates the number and sizes of connected components."""
        components = list(nx.connected_components(self.nx_graph))
        component_sizes = [len(component) for component in components]
        component_counts = Counter(component_sizes)
        return component_counts

    def centrality_distributions(self) -> Dict[str, Dict[float, float]]:
        """Calculates degree, betweenness, and closeness centrality distributions."""
        degree_centrality = nx.degree_centrality(self.nx_graph)
        betweenness_centrality = nx.betweenness_centrality(self.nx_graph)
        closeness_centrality = nx.closeness_centrality(self.nx_graph)

        degree_counts = Counter(degree_centrality.values())
        betweenness_counts = Counter(betweenness_centrality.values())
        closeness_counts = Counter(closeness_centrality.values())

        total_nodes = len(self.nx_graph.nodes)
        return {
            "degree": {
                centrality: count / total_nodes
                for centrality, count in degree_counts.items()
            },
            "betweenness": {
                centrality: count / total_nodes
                for centrality, count in betweenness_counts.items()
            },
            "closeness": {
                centrality: count / total_nodes
                for centrality, count in closeness_counts.items()
            },
        }

    def diameter(self) -> int:
        """Calculates the diameter of the graph."""
        if not nx.is_connected(self.nx_graph):
            raise ValueError("Graph is not connected. Cannot compute diameter.")
        return nx.diameter(self.nx_graph)

    def assortativity(self) -> float:
        """Calculates the degree assortativity coefficient."""
        return nx.degree_assortativity_coefficient(self.nx_graph)

    def full_metrics_plot(self):
        """Plots various metrics and distributions in the specified order."""
        # Получаем все необходимые данные
        degree_dist = self.degree_distribution()
        shortest_path_dist = self.shortest_path_length_distribution()
        clustering_dist = self.clustering_coefficient_distribution()
        centrality_dist = self.centrality_distributions()
        components = self.connected_components()

        # Создаем фигуру с 3 строками и соответствующими столбцами
        fig, axs = plt.subplots(3, 3, figsize=(18, 15))

        # Первая строка: degree_dist, shortest_path_dist, clustering_dist
        self._plot_distribution(axs[0, 0], degree_dist, "Degree Distribution", "Degree", "Frequency", "skyblue")
        self._plot_distribution(axs[0, 1], shortest_path_dist, "Shortest Path Length Distribution", "Path Length", "Frequency", "salmon")
        self._plot_distribution(axs[0, 2], clustering_dist, "Clustering Coefficient Distribution", "Clustering Coefficient", "Frequency", "lightgreen")

        # Вторая строка: centrality_dist (три графика)
        centrality_colors = {"degree": "orange", "betweenness": "purple", "closeness": "brown"}
        for i, (centrality_type, dist) in enumerate(centrality_dist.items()):
            self._plot_distribution(axs[1, i], dist, f"{centrality_type.capitalize()} Centrality", "Centrality Value", "Frequency", centrality_colors[centrality_type])

        # Третья строка: connected components и текстовые метрики
        self._plot_distribution(axs[2, 0], components, "Connected Components Sizes", "Component Size", "Count", "teal")

        # Текстовое поле с другими метриками
        try:
            diameter = self.diameter()
        except ValueError:
            diameter = "N/A (graph is disconnected)"
        assortativity = self.assortativity()

        axs[2, 1].text(
            0.1, 0.5,
            f"Diameter: {diameter}\n"
            f"Assortativity: {assortativity:.2f}\n"
            f"Number of Components: {sum(components.values())}",
            fontsize=12, bbox=dict(facecolor='white', alpha=0.5)
        )
        axs[2, 1].axis('off')

        # Удаляем пустой график в третьей строке
        axs[2, 2].axis('off')

        plt.tight_layout()
        plt.show()

    def _plot_distribution(self, ax, data: Dict[float, float], title: str, xlabel: str, ylabel: str, color: str):
        """Helper function to plot a distribution."""
        if not data:
            ax.text(0.5, 0.5, "No data available", fontsize=12, ha="center", va="center")
            ax.axis('off')
            return

        keys, values = zip(*sorted(data.items()))
        ax.bar(keys, values, color=color, edgecolor="black", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)