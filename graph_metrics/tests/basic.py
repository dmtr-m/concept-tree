import os
import sys
import inspect

import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)

from directed_graph.graph import Graph
from graph_metrics.metrics import GraphMetrics

my_graph = Graph()
my_graph.add_vertex("A")
my_graph.add_vertex("B")
my_graph.add_vertex("C")
my_graph.add_edge("A", "B", "rel1")
my_graph.add_edge("B", "C", "rel2")
my_graph.add_edge("C", "A", "rel3")
my_graph.add_edge("A", "C", "rel4")

# Calculate and print metrics
metrics = GraphMetrics(my_graph)

print(f"Average Degree: {metrics.average_degree()}")
print(f"Degree Distribution: {metrics.degree_distribution()}")
print(f"Shortest Path (A to C): {metrics.shortest_path_length('A', 'C')}")
print(f"Average Shortest Path Length: {metrics.average_shortest_path_length()}")
print(f"Diameter: {metrics.diameter()}")
print(f"Clustering Coefficient (A): {metrics.clustering_coefficient('A')}")
print(f"Average Clustering Coefficient: {metrics.average_clustering_coefficient()}")
print(f"Assortativity: {metrics.assortativity()}")