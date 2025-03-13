import unittest

import sys
import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from directed_graph.graph import Graph

from fuzzy_rdf_match import FuzzyRDFMatcher

def test_triangle():
    data_graph = Graph()
    data_graph.add_vertex("A")
    data_graph.add_vertex("B")
    data_graph.add_vertex("C")
    data_graph.add_vertex("D")

    data_graph.add_edge("B", "A", "a")
    data_graph.add_edge("B", "D", "b")
    data_graph.add_edge("D", "A", "c")
    data_graph.add_edge("B", "C", "c")
    data_graph.add_edge("C", "A", "c")

    query_graph = Graph()
    query_graph.add_vertex("X")
    query_graph.add_vertex("Y")
    query_graph.add_vertex("Z")

    query_graph.add_edge("Y", "X", "a")
    query_graph.add_edge("Y", "Z", "c")
    query_graph.add_edge("Z", "X", "c")

    print("-" * 40)
    print("Expect only one match")

    matcher = FuzzyRDFMatcher(data_graph, precision=0.95)
    matched_subgraphs = matcher.match(query_graph)
    for subgraph in matched_subgraphs:
        print(subgraph)

def test_full_match():
    data_graph = Graph()
    data_graph.add_vertex("A")
    data_graph.add_vertex("B")
    data_graph.add_vertex("C")
    data_graph.add_vertex("D")

    data_graph.add_edge("B", "A", "a")
    data_graph.add_edge("B", "D", "b")
    data_graph.add_edge("D", "A", "c")
    data_graph.add_edge("B", "C", "c")
    data_graph.add_edge("C", "A", "c")

    query_graph = Graph()
    query_graph.add_vertex("X")
    query_graph.add_vertex("Y")
    query_graph.add_vertex("Z")
    query_graph.add_vertex("W")

    query_graph.add_edge("Y", "X", "a")
    query_graph.add_edge("Y", "Z", "c")
    query_graph.add_edge("Z", "X", "c")
    query_graph.add_edge("Y", "W", "b")
    query_graph.add_edge("W", "X", "c")

    print("-" * 40)
    print("Expect only one match")

    matcher = FuzzyRDFMatcher(data_graph, precision=0.95)
    matched_subgraphs = matcher.match(query_graph)
    for subgraph in matched_subgraphs:
        print(subgraph)

def test_triangle_low_threshold():
    data_graph = Graph()
    data_graph.add_vertex("A")
    data_graph.add_vertex("B")
    data_graph.add_vertex("C")
    data_graph.add_vertex("D")

    data_graph.add_edge("B", "A", "a")
    data_graph.add_edge("B", "D", "b")
    data_graph.add_edge("D", "A", "c")
    data_graph.add_edge("B", "C", "c")
    data_graph.add_edge("C", "A", "c")

    query_graph = Graph()
    query_graph.add_vertex("X")
    query_graph.add_vertex("Y")
    query_graph.add_vertex("Z")

    query_graph.add_edge("Y", "X", "a")
    query_graph.add_edge("Y", "Z", "c")
    query_graph.add_edge("Z", "X", "c")

    print("-" * 40)
    print("Expect two matches")

    matcher = FuzzyRDFMatcher(data_graph, precision=0.7)
    matched_subgraphs = matcher.match(query_graph)
    for subgraph in matched_subgraphs:
        print(subgraph)

def main():
    test_triangle()
    test_full_match()
    test_triangle_low_threshold()

if __name__ == "__main__":
    main()
