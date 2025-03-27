import os
import sys
import numpy as np
from typing import List, Dict, Union
from collections import Counter, defaultdict
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
        Vertices with the same vertex_type are merged into one logical vertex.
        Edges with the same edge_type are merged into one logical edge.
        """
        nx_graph = nx.Graph()

        for edge in graph.get_edges():
            nx_graph.add_node(edge.agent_1)
            nx_graph.add_node(edge.agent_2)
            nx_graph.add_edge(edge.agent_1, edge.agent_2)

        # # Group vertices by their vertex_type
        # vertex_type_map = defaultdict(list)
        # for vertex in graph.vertices.values():
        #     vertex_type_map[vertex.vertex_type].append(vertex.concept)

        # # Add merged vertices (logical nodes)
        # for vertex_type, concepts in vertex_type_map.items():
        #     merged_vertex_name = f"Type_{vertex_type}"
        #     nx_graph.add_node(merged_vertex_name, concepts=concepts)

        # # Group edges by their edge_type
        # edge_type_map = defaultdict(set)
        # for edge in graph.get_edges():
        #     agent_1 = graph.vertices[edge.agent_1]
        #     agent_2 = graph.vertices[edge.agent_2]

        #     merged_vertex_1 = f"Type_{agent_1.vertex_type}"
        #     merged_vertex_2 = f"Type_{agent_2.vertex_type}"

        #     if merged_vertex_1 != merged_vertex_2:  # Avoid self-loops
        #         edge_key = (merged_vertex_1, merged_vertex_2, edge.edge_type)
        #         edge_type_map[edge_key].add(edge.label)

        # # Add merged edges (logical edges)
        # for (vertex_1, vertex_2, edge_type), labels in edge_type_map.items():
        #     merged_edge_label = ", ".join(
        #         sorted(labels)
        #     )  # Combine all labels for the same edge_type
        #     nx_graph.add_edge(
        #         vertex_1, vertex_2, label=merged_edge_label, edge_type=edge_type
        #     )

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
        self._plot_distribution(
            axs[0, 0],
            degree_dist,
            "Degree Distribution",
            "Degree",
            "Frequency",
            "skyblue",
        )
        self._plot_distribution(
            axs[0, 1],
            shortest_path_dist,
            "Shortest Path Length Distribution",
            "Path Length",
            "Frequency",
            "salmon",
        )
        self._plot_distribution(
            axs[0, 2],
            clustering_dist,
            "Clustering Coefficient Distribution",
            "Clustering Coefficient",
            "Frequency",
            "lightgreen",
        )

        # Вторая строка: centrality_dist (три графика)
        centrality_colors = {
            "degree": "orange",
            "betweenness": "purple",
            "closeness": "brown",
        }
        for i, (centrality_type, dist) in enumerate(centrality_dist.items()):
            self._plot_distribution(
                axs[1, i],
                dist,
                f"{centrality_type.capitalize()} Centrality",
                "Centrality Value",
                "Frequency",
                centrality_colors[centrality_type],
            )

        # Третья строка: connected components и текстовые метрики
        self._plot_distribution(
            axs[2, 0],
            components,
            "Connected Components Sizes",
            "Component Size",
            "Count",
            "teal",
        )

        # Текстовое поле с другими метриками
        try:
            diameter = self.diameter()
        except ValueError:
            diameter = "N/A (graph is disconnected)"
        assortativity = self.assortativity()

        axs[2, 1].text(
            0.1,
            0.5,
            f"Diameter: {diameter}\n"
            f"Assortativity: {assortativity:.2f}\n"
            f"Number of Components: {sum(components.values())}",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )
        axs[2, 1].axis("off")

        # Удаляем пустой график в третьей строке
        axs[2, 2].axis("off")

        plt.tight_layout()
        plt.show()

    def _plot_distribution(
        self,
        ax,
        data: Dict[float, float],
        title: str,
        xlabel: str,
        ylabel: str,
        color: str,
    ):
        """Helper function to plot a distribution."""
        if not data:
            ax.text(
                0.5, 0.5, "No data available", fontsize=12, ha="center", va="center"
            )
            ax.axis("off")
            return

        keys, values = zip(*sorted(data.items()))
        ax.bar(keys, values, color=color, edgecolor="black", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)

    def detect_hubs_with_metrics(
        self,
        degree_weight: float = 0.4,
        betweenness_weight: float = 0.3,
        closeness_weight: float = 0.3,
        threshold: float = None,
        top_n: int = None,
    ) -> List[str]:
        """
        Detects hub vertices in the graph based on a combination of metrics.

        Args:
            Same as before.

        Returns:
            List[str]: List of hub vertex names.
        """
        if threshold is not None and top_n is not None:
            raise ValueError("Only one of 'threshold' or 'top_n' can be specified.")

        # Calculate individual metrics
        degree_centrality = nx.degree_centrality(self.nx_graph)
        betweenness_centrality = nx.betweenness_centrality(self.nx_graph)
        closeness_centrality = nx.closeness_centrality(self.nx_graph)

        # Normalize weights
        total_weight = degree_weight + betweenness_weight + closeness_weight
        degree_weight /= total_weight
        betweenness_weight /= total_weight
        closeness_weight /= total_weight

        # Compute composite score for each node
        composite_scores = {}
        for node in self.nx_graph.nodes:
            degree_score = degree_centrality[node]
            betweenness_score = betweenness_centrality[node]
            closeness_score = closeness_centrality[node]

            # Assortativity contribution is global, so we distribute it evenly
            composite_score = (
                degree_weight * degree_score
                + betweenness_weight * betweenness_score
                + closeness_weight * closeness_score
            )
            composite_scores[node] = composite_score

        # Select hubs based on threshold or top_n
        if threshold is not None:
            hubs = [
                node for node, score in composite_scores.items() if score >= threshold
            ]
        elif top_n is not None:
            sorted_nodes = sorted(
                composite_scores.items(), key=lambda x: x[1], reverse=True
            )
            hubs = [node for node, _ in sorted_nodes[:top_n]]
        else:
            raise ValueError("Either 'threshold' or 'top_n' must be specified.")

        return hubs

    def get_hub_labels(self, hubs: List[str]) -> Dict[str, str]:
        """
        Retrieves labels for the given hub nodes.

        Args:
            hubs (List[str]): List of hub node names.

        Returns:
            Dict[str, str]: Dictionary mapping hub names to their labels.
        """
        hub_labels = {}
        for hub in hubs:
            # Get the label from node attributes, defaulting to the node name itself
            label = self.nx_graph.nodes[hub].get("label", hub)
            hub_labels[hub] = label
        return hub_labels

    def plot_hub_detection_with_metrics_and_labels(
        self,
        degree_weight: float = 0.4,
        betweenness_weight: float = 0.3,
        closeness_weight: float = 0.3,
        threshold: float = None,
        top_n: int = None,
    ):
        """
        Plots the graph with detected hubs highlighted and displays their labels.

        Args:
            Same as `detect_hubs_with_metrics`.
        """
        hubs = self.detect_hubs_with_metrics(
            degree_weight=degree_weight,
            betweenness_weight=betweenness_weight,
            closeness_weight=closeness_weight,
            threshold=threshold,
            top_n=top_n,
        )
        hub_set = set(hubs)
        hub_labels = self.get_hub_labels(hubs)

        # Draw the graph
        pos = nx.spring_layout(self.nx_graph, seed=42)  # Layout for visualization
        plt.figure(figsize=(10, 8))

        # Draw non-hub nodes
        non_hub_nodes = [node for node in self.nx_graph.nodes if node not in hub_set]
        nx.draw_networkx_nodes(
            self.nx_graph,
            pos,
            nodelist=non_hub_nodes,
            node_color="lightblue",
            node_size=300,
            label="Non-Hub Nodes",
        )

        # Draw hub nodes
        nx.draw_networkx_nodes(
            self.nx_graph,
            pos,
            nodelist=hubs,
            node_color="red",
            node_size=500,
            label="Hub Nodes",
        )

        # Draw edges
        nx.draw_networkx_edges(self.nx_graph, pos, alpha=0.5, edge_color="gray")

        # Draw labels for all nodes
        all_labels = {
            node: self.nx_graph.nodes[node].get("label", node)
            for node in self.nx_graph.nodes
        }
        nx.draw_networkx_labels(
            self.nx_graph, pos, labels=all_labels, font_size=10, font_color="black"
        )

        # Highlight hub labels
        for hub, label in hub_labels.items():
            x, y = pos[hub]
            plt.text(
                x,
                y + 0.05,
                label,
                fontsize=9,
                ha="center",
                color="darkred",
                bbox=dict(facecolor="white", alpha=0.7),
            )

        plt.title("Graph with Detected Hubs and Labels")
        plt.legend(loc="upper right")
        plt.axis("off")
        plt.show()
