import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

from directed_graph.graph import Graph, save_graph, load_graph

def test_save():
    graph = Graph()

    graph.add_vertex("cat")
    graph.add_vertex("dog")
    graph.add_edge("dog", "cat", "bark")

    graph.add_vertex("orange cat", ["orange", "cat"])
    graph.add_edge("orange cat", "cat", "follow")

    graph.add_vertex("black dog", ["black", "dog"])
    graph.add_edge("black dog", "orange cat", "fear", 1, 3)

    save_graph(graph, "graph.pickle")


def test_load():
    graph = load_graph("graph.pickle")

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


if __name__ == "__main__":
    test_save()
    test_load()
