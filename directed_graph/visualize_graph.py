import plotly.graph_objects as go
from directed_graph.graph import Graph
from directed_graph.vertex import Vertex
import numpy as np

def visualize_graph_3d(graph: Graph):
    """
    Visualize a graph in 3D space using Plotly.
    """
    # Extract vertex coordinates
    x_nodes, y_nodes, z_nodes = [], [], []
    node_labels = []
    
    for concept, vertex in graph.vertices.items():
        if vertex.embedding.shape[0] < 3:
            raise ValueError("Vertex embeddings must be at least 3D for visualization.")
        x_nodes.append(vertex.embedding[0])
        y_nodes.append(vertex.embedding[1])
        z_nodes.append(vertex.embedding[2])
        node_labels.append(concept)
    
    # Extract edge coordinates and labels
    edge_x, edge_y, edge_z = [], [], []
    edge_labels, edge_label_x, edge_label_y, edge_label_z = [], [], [], []
    for edge in graph.edges:
        v1, v2 = graph.vertices[edge.agent_1], graph.vertices[edge.agent_2]
        edge_x.extend([v1.embedding[0], v2.embedding[0], None])
        edge_y.extend([v1.embedding[1], v2.embedding[1], None])
        edge_z.extend([v1.embedding[2], v2.embedding[2], None])
        
        # Add label in the middle of the edge
        edge_label_x.append((v1.embedding[0] + v2.embedding[0]) / 2)
        edge_label_y.append((v1.embedding[1] + v2.embedding[1]) / 2)
        edge_label_z.append((v1.embedding[2] + v2.embedding[2]) / 2)
        edge_labels.append(edge.label)
    
    # Create the edge traces
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(width=2, color='gray'),
        name='Edges'
    )
    
    # Create the node traces
    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers+text',
        marker=dict(size=8, color='blue', opacity=0.8),
        text=node_labels,
        textposition='top center',
        name='Nodes'
    )
    
    # Create edge label trace
    edge_label_trace = go.Scatter3d(
        x=edge_label_x, y=edge_label_y, z=edge_label_z,
        mode='text',
        text=edge_labels,
        textposition='middle center',
        name='Edge Labels'
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace])
    fig.update_layout(
        title="3D Graph Visualization",
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis"
        ),
        showlegend=False
    )

    fig.update_layout(
        width=1000,  # Width in pixels
        height=1000  # Height in pixels
    )
    
    fig.show()

def visualize_graph_ngrams(graph: Graph):
    """
    Visualize a graph in 3D space using Plotly.
    """

    def z_coordinate(vertex: Vertex) -> float:
        return (len(vertex.words_of_concept) - 1) / 2.5
    
    def colour(vertex: Vertex) -> str:
        colours = ["red", "green", "blue"]
        return colours[len(vertex.words_of_concept) - 1]

    # Extract vertex coordinates
    x_nodes, y_nodes, z_nodes = [], [], []
    node_labels = []
    colours = []
    
    for concept, vertex in graph.vertices.items():
        if vertex.embedding.shape[0] < 3:
            raise ValueError("Vertex embeddings must be at least 3D for visualization.")
        x_nodes.append(vertex.embedding[0])
        y_nodes.append(vertex.embedding[1])
        z_nodes.append(z_coordinate(vertex))
        node_labels.append(concept)
        colours.append(colour(vertex))
    
    # Extract edge coordinates and labels
    edge_x, edge_y, edge_z = [], [], []
    edge_labels, edge_label_x, edge_label_y, edge_label_z = [], [], [], []
    for edge in graph.edges:
        v1, v2 = graph.vertices[edge.agent_1], graph.vertices[edge.agent_2]
        edge_x.extend([v1.embedding[0], v2.embedding[0], None])
        edge_y.extend([v1.embedding[1], v2.embedding[1], None])
        edge_z.extend([z_coordinate(v1), z_coordinate(v2), None])
        
        # Add label in the middle of the edge
        edge_label_x.append((v1.embedding[0] + v2.embedding[0]) / 2)
        edge_label_y.append((v1.embedding[1] + v2.embedding[1]) / 2)
        edge_label_z.append((z_coordinate(v1) + z_coordinate(v2)) / 2)
        edge_labels.append(edge.label)
    
    # Create the edge traces
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(width=2, color='gray'),
        name='Edges'
    )
    
    # Create the node traces
    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers+text',
        marker=dict(size=8, color=colours, opacity=0.8),
        text=node_labels,
        textposition='top center',
        name='Nodes'
    )
    
    # Create edge label trace
    edge_label_trace = go.Scatter3d(
        x=edge_label_x, y=edge_label_y, z=edge_label_z,
        mode='text',
        text=edge_labels,
        textposition='middle center',
        name='Edge Labels'
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace])
    fig.update_layout(
        title="3D Graph Visualization",
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis"
        ),
        showlegend=False
    )

    fig.update_layout(
        width=1000,   # Width in pixels
        height=1000,  # Height in pixels
        # scene=dict(
        #     xaxis=dict(range=[-.032, .025]),  # X-axis limits
        #     yaxis=dict(range=[-.032, .025]), # Y-axis limits
        #     zaxis=dict(range=[-.05, 1])  # Z-axis limits
        # )
    )
    
    fig.show()
