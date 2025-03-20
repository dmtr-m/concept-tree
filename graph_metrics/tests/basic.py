import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)

from directed_graph.graph import Graph, visualize_graph
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
metrics.full_metrics_plot()

visualize_graph(my_graph)

