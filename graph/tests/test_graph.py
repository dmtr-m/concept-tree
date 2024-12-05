import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from graph import Graph, visualize_graph

def test_sentences():
    graph = Graph()

    graph.add_vertex("ускорение свободного падения", ["ускорение", "свободный", "падение"])
    graph.add_vertex("высота", ["Высота"])
    graph.add_edge("ускорение свободного падения", "высота", "зависеть", 1, 1)

    graph.add_vertex("потенциальная энергия", ["потенциальный", "энергия"])
    graph.add_edge("потенциальная энергия", "высота", "зависеть", 1, 2)

    graph.add_vertex("скорость")
    graph.add_edge("высота", "скорость", "зависеть", 1, 3)

    print("Graph structure:")
    print(graph)
    print("\nDetailed representation:")
    print(repr(graph))

    print("\nEdges:")
    for concept in graph.vertices.keys():
        print(f"Concept: {concept}")
        print(graph.get_vertex_edges(concept))

    print("\nVertices:")
    for concept in graph.vertices.keys():
        print(graph.vertices[concept])

    visualize_graph(graph)

if __name__ == "__main__":
    test_sentences()
