import sys
import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from typing import List, Dict, Tuple
import itertools

from directed_graph.vertex import Vertex
from directed_graph.edge import Edge
from directed_graph.graph import Graph


class GraphDecomposer:
    """Decomposes graphs into a list of paths"""

    @staticmethod
    def decompose_full_graph(graph: Graph) -> Tuple[List[List[str]], List[List[Edge]]]:
        """
        Decompose the data graph into all possible simple paths.

        Returns:
            Tuple:
                - List[List[str]]: All vertex paths.
                - List[List[Edge]]: All edge paths.
        """
        graph_vertex_paths = []
        graph_edge_paths = []

        for start_vertex in graph.vertices:
            for end_vertex in graph.vertices:
                if start_vertex != end_vertex:
                    new_vertex_paths, new_edge_paths = GraphDecomposer.find_all_paths(
                        graph, start_vertex, end_vertex
                    )
                    graph_vertex_paths.extend(new_vertex_paths)
                    graph_edge_paths.extend(new_edge_paths)

        return graph_vertex_paths, graph_edge_paths

    @staticmethod
    def decompose_with_root(query_graph: Graph) -> Tuple[List[List[str]], List[List[Edge]]]:
        """
        Decompose the query graph into simple paths from roots to destinations.

        Root vertices are those with no incoming edges. If none are found, the vertex with the highest
        (out-degree - in-degree) is chosen. Destination vertices are those with no outgoing edges.

        Args:
            query_graph (Graph): The query graph to decompose.

        Returns:
            Tuple:
                - List[List[str]]: Vertex paths.
                - List[List[Edge]]: Edge paths.
        """
        vertex_paths = []
        edge_paths = []

        roots = [v for v in query_graph.vertices if not query_graph.get_incoming_edges(v)]
        if not roots:
            sorted_vertices = sorted(
                query_graph.vertices,
                key=lambda v: len(query_graph.get_outgoing_edges(v)) - len(query_graph.get_incoming_edges(v)),
                reverse=True,
            )
            roots = [sorted_vertices[0]]

        destinations = [v for v in query_graph.vertices if not query_graph.get_outgoing_edges(v)]
        if not destinations:
            destinations = query_graph.vertices

        for root in roots:
            for dest in destinations:
                new_vertex_paths, new_edge_paths = GraphDecomposer.find_all_paths(query_graph, root, dest)
                vertex_paths.extend(new_vertex_paths)
                edge_paths.extend(new_edge_paths)

        return vertex_paths, edge_paths

    @staticmethod
    def find_all_paths(
        graph: Graph, start: str, end: str, vertex_path: List[str] = None, edge_path: List[Edge] = None
    ) -> Tuple[List[List[str]], List[List[Edge]]]:
        """
        Recursively find all simple paths from start to end.

        Args:
            graph (Graph): The graph to search.
            start (str): The starting vertex.
            end (str): The destination vertex.
            vertex_path (List[str], optional): Current vertex path. Defaults to None.
            edge_path (List[Edge], optional): Current edge path. Defaults to None.

        Returns:
            Tuple:
                - List[List[str]]: All vertex paths.
                - List[List[Edge]]: All edge paths.
        """
        if vertex_path is None:
            vertex_path = [start]
            edge_path = []

        if start == end:
            return [vertex_path], [edge_path]

        all_vertex_paths = []
        all_edge_paths = []
        for edge in graph.get_outgoing_edges(start):
            next_vertex = edge.agent_2
            if next_vertex not in vertex_path:
                new_vertex_paths, new_edge_paths = GraphDecomposer.find_all_paths(
                    graph, next_vertex, end, vertex_path + [next_vertex], edge_path + [edge]
                )
                all_vertex_paths.extend(new_vertex_paths)
                all_edge_paths.extend(new_edge_paths)

        return all_vertex_paths, all_edge_paths


class EditDistanceCalculator:
    """Calculates edit distance for give paths"""

    SUBSTITUTION_COST_EQUAL = 0
    SUBSTITUTION_COST_NEQUAL = 1
    DELETION_COST = 0.5
    INSERTION_COST = 0.5

    @staticmethod
    def calculate(query_path: List[Edge], matched_path: List[Edge]) -> float:
        """
        Compute the edit distance between two sequences of edge labels.

        Costs:
            - Substitution: 0 if labels match, 1 otherwise.
            - Insertion and Deletion: 0.5 each.

        Args:
            path1 (List[Edge]): First path.
            path2 (List[Edge]): Second path.

        Returns:
            float: The edit distance between the two paths.
        """
        seq1 = [v.label for v in query_path]
        seq2 = [v.label for v in matched_path]
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i * EditDistanceCalculator.DELETION_COST
        for j in range(n + 1):
            dp[0][j] = j * EditDistanceCalculator.INSERTION_COST

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = EditDistanceCalculator.SUBSTITUTION_COST_EQUAL if seq1[i - 1] == seq2[j - 1] else EditDistanceCalculator.SUBSTITUTION_COST_NEQUAL
                dp[i][j] = min(
                    dp[i - 1][j] + EditDistanceCalculator.DELETION_COST,
                    dp[i][j - 1] + EditDistanceCalculator.INSERTION_COST,
                    dp[i - 1][j - 1] + cost,
                )
        
        dist = dp[m][n] / len(query_path)
        return dist


class GraphEvaluator:
    """Evaluates precision of isomorphism search"""

    TOO_MANY_VERTICES_THRESHOLD = 0.1

    @staticmethod
    def evaluate(query_graph: Graph, found_subgraph: Graph) -> float:
        """
        Evaluate similarity between the query and found graphs.

        Args:
            query_graph (Graph): The query graph.
            found_graph (Graph): The subgraph that was found during previous steps.
        
        Returns:
            float: Similarity degree.
        """
        if (len(query_graph.vertices) - len(found_subgraph.vertices)) / len(query_graph.vertices) > GraphEvaluator.TOO_MANY_VERTICES_THRESHOLD:
            return 0
        
        return 1


class FuzzyRDFMatcher:
    """Fuzzy RDF Graph Matcher using path decomposition and edit distance."""

    def __init__(self, graph: Graph, precision: float = 0.5):
        """
        Initialize the matcher with a fuzzy RDF graph.

        Args:
            graph (Graph): The input data graph.
            precision (float, optional): A percentage of a required match satisfaction degree for
                a match. Defaults to 0.5.
        """
        self.data_graph = graph
        self.precision = precision
        self.vertex_paths, self.edge_paths = GraphDecomposer.decompose_full_graph(self.data_graph)

    def find_paths_candidates(self, query_path: List[Edge]) -> List[List[Edge]]:
        """
        Find candidate data paths that are similar to the query path.

        Args:
            query_path (List[Edge]): The query path.
            threshold (float, optional): Maximum allowed edit distance. Defaults to 0.5.

        Returns:
            List[List[Edge]]: Valid matching paths.
        """
        valid_paths = []
        for data_path in self.edge_paths:
            if abs(len(query_path) - len(data_path)) / len(query_path) <= 1 - self.precision:
                dist = EditDistanceCalculator.calculate(query_path, data_path)
                if dist <= 1 - self.precision:
                    valid_paths.append(data_path)
        return valid_paths
    
    def join_candidate_paths(self, candidate_paths: Dict[Tuple[str], List[Edge]]) -> List[Graph]:
        """
        Join candidate paths into possible matching subgraphs.

        Args:
            canditate_paths (Dict[Tuple[str], List[Edge]]): A dictionary containing
                paths from the query graph and similar paths from the data graph.
        
        Returns:
            List[Graph]: A list of subgraphs that were created by matching candidate paths.
        """
        joined_subgraphs = []
        for combo in itertools.product(*candidate_paths.values()):
            edges = list()
            for path in combo:
                edges.extend(path)

            subgraph = Graph.build_from_vertices_and_edges(edges=list(edges))
            joined_subgraphs.append(subgraph)
        return joined_subgraphs

    def match(self, query_graph: Graph) -> List[Graph]:
        """
        Match a query graph against the data graph.

        Args:
            query_graph (Graph): The query graph.

        Returns:
            List[Graph]: Valid matching subgraphs.
        """
        query_vertex_paths, query_edge_paths = GraphDecomposer.decompose_with_root(query_graph)
        if not query_vertex_paths:
            return []

        path_matches = dict()
        for query_edge_path in query_edge_paths:
            key = tuple(edge.label for edge in query_edge_path)
            path_matches[key] = self.find_paths_candidates(query_edge_path)

        matched_subgraphs = self.join_candidate_paths(path_matches)
        
        valid_subgraphs = []
        for subgraph in matched_subgraphs:
            if GraphEvaluator.evaluate(query_graph, subgraph) > 0.5:
                valid_subgraphs.append(subgraph)
        
        return valid_subgraphs

if __name__ == "__main__":
    print("To test this masterpiece, feel free to go to the tests files")
