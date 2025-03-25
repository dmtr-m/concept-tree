from graph.higher_dim_graph import Graph
from graph.graph import visualize_graph
from graph.edge import Edge

def test_simple_higher_dim_graph():
    graph = Graph()
    
    graph.add_vertex("A")
    graph.add_vertex("B")
    graph.add_vertex("C")
    
    graph.add_union_edge([Edge("A", "B", "and", 1, 1)], "C", "is", 1)
    
    visualize_graph(graph)

def test_classic_higher_dim_graph():
    graph = Graph()
    
    graph.add_vertex("Мальчик")
    graph.add_vertex("Девочка")
    graph.add_vertex("Кино")

    graph.add_edge("Мальчик", "Девочка", "позвонил", 1, 1)
    graph.add_union_edge([Edge("Мальчик", "Девочка", "и", 1, 1)], "Кино", "пойти", 1)
    
    visualize_graph(graph)

if __name__ == "__main__":
    test_classic_higher_dim_graph()
