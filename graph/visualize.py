import plotly.graph_objects as go
from directed_graph.graph import Graph
import numpy as np
from sklearn.decomposition import PCA
from collections import defaultdict


def visualize_graph_3d(graph: Graph):
    """
    Visualize a graph in 3D space using Plotly.
    """
    # Extract vertex coordinates
    x_nodes, y_nodes, z_nodes = [], [], []
    node_labels = []

    for concept, data in graph.get_all_vertices():
        if data["embedding"].shape[0] < 3:
            raise ValueError("Vertex embeddings must be at least 3D for visualization.")
        x_nodes.append(data["embedding"][0])
        y_nodes.append(data["embedding"][1])
        z_nodes.append(data["embedding"][2])
        node_labels.append(concept)

    # Extract edge coordinates and labels
    edge_x, edge_y, edge_z = [], [], []
    edge_labels, edge_label_x, edge_label_y, edge_label_z = [], [], [], []
    for from_label, to_label, key, edge_data in graph.get_all_edges():
        v1, v2 = graph.graph.nodes[from_label], graph.graph.nodes[to_label]
        edge_x.extend([v1["embedding"][0], v2["embedding"][0], None])
        edge_y.extend([v1["embedding"][1], v2["embedding"][1], None])
        edge_z.extend([v1["embedding"][2], v2["embedding"][2], None])

        # Add label in the middle of the edge
        edge_label_x.append((v1["embedding"][0] + v2["embedding"][0]) / 2)
        edge_label_y.append((v1["embedding"][1] + v2["embedding"][1]) / 2)
        edge_label_z.append((v1["embedding"][2] + v2["embedding"][2]) / 2)
        edge_labels.append(edge_data["label"])

    # Create the edge traces
    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(width=2, color="gray"),
        name="Edges",
    )

    # Create the node traces
    node_trace = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode="markers+text",
        marker=dict(size=8, color="blue", opacity=0.8),
        text=node_labels,
        textposition="top center",
        name="Nodes",
    )

    # Create edge label trace
    edge_label_trace = go.Scatter3d(
        x=edge_label_x,
        y=edge_label_y,
        z=edge_label_z,
        mode="text",
        text=edge_labels,
        textposition="middle center",
        name="Edge Labels",
    )

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace])
    fig.update_layout(
        title="3D Graph Visualization",
        scene=dict(xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis"),
        showlegend=False,
    )

    fig.update_layout(width=1000, height=1000)  # Width in pixels  # Height in pixels

    fig.show()


def visualize_graph_ngrams(graph: Graph):
    """
    Visualize a graph in 3D space using Plotly.
    """

    def z_coordinate(data: dict) -> float:
        return (len(data["words_of_label"]) - 1) / 2.5

    def colour(data: dict) -> str:
        colours = ["red", "green", "blue"]
        return colours[min(len(data["words_of_label"]), len(colours)) - 1]

    # Extract vertex coordinates
    x_nodes, y_nodes, z_nodes = [], [], []
    node_labels = []
    colours = []

    for concept, data in graph.get_all_vertices():
        if data["embedding"].shape[0] < 3:
            raise ValueError("Vertex embeddings must be at least 3D for visualization.")
        x_nodes.append(data["embedding"][0])
        y_nodes.append(data["embedding"][1])
        z_nodes.append(z_coordinate(data))
        node_labels.append(concept)
        colours.append(colour(data))

    # Extract edge coordinates and labels
    edge_x, edge_y, edge_z = [], [], []
    edge_labels, edge_label_x, edge_label_y, edge_label_z = [], [], [], []
    for from_label, to_label, key, edge_data in graph.get_all_edges():
        v1, v2 = graph.graph.nodes[from_label], graph.graph.nodes[to_label]
        edge_x.extend([v1["embedding"][0], v2["embedding"][0], None])
        edge_y.extend([v1["embedding"][1], v2["embedding"][1], None])
        edge_z.extend([z_coordinate(v1), z_coordinate(v2), None])

        # Add label in the middle of the edge
        edge_label_x.append((v1["embedding"][0] + v2["embedding"][0]) / 2)
        edge_label_y.append((v1["embedding"][1] + v2["embedding"][1]) / 2)
        edge_label_z.append((z_coordinate(v1) + z_coordinate(v2)) / 2)
        edge_labels.append(edge_data["label"])

    # Create the edge traces
    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(width=2, color="gray"),
        name="Edges",
    )

    # Create the node traces
    node_trace = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode="markers+text",
        marker=dict(size=8, color=colours, opacity=0.8),
        text=node_labels,
        textposition="top center",
        name="Nodes",
    )

    # Create edge label trace
    edge_label_trace = go.Scatter3d(
        x=edge_label_x,
        y=edge_label_y,
        z=edge_label_z,
        mode="text",
        text=edge_labels,
        textposition="middle center",
        name="Edge Labels",
    )

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace])
    fig.update_layout(
        title="3D Graph Visualization",
        scene=dict(xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis"),
        showlegend=False,
    )

    fig.update_layout(
        width=1000,  # Width in pixels
        height=1000,  # Height in pixels
        # scene=dict(
        #     xaxis=dict(range=[-.032, .025]),  # X-axis limits
        #     yaxis=dict(range=[-.032, .025]), # Y-axis limits
        #     zaxis=dict(range=[-.05, 1])  # Z-axis limits
        # )
    )
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )

    fig.show()


def visualize_graph_ngrams_with_pca(graph: Graph):
    """
    Visualize a graph in 3D space using Plotly with PCA for x and y coordinates,
    applied separately for each z-layer.
    """

    def z_coordinate(data: dict) -> float:
        return (len(data["words_of_label"]) - 1) / 2.5

    def colour(data: dict) -> str:
        colours = ["red", "green", "blue"]
        return colours[min(len(data["words_of_label"]), len(colours)) - 1]

    # Group vertices by their z-coordinate
    z_groups = defaultdict(list)
    for concept, data in graph.get_all_vertices():
        z = z_coordinate(data)
        z_groups[z].append((concept, data))

    # Apply PCA to each group of vertices
    reduced_embeddings = {}
    excluded_vertices = set()  # To track excluded vertices
    for z, vertices in z_groups.items():
        embeddings = [data["embedding"] for _, data in vertices]

        # Check dimensions and filter out inconsistent embeddings
        if len(embeddings) == 0:
            continue
        dims = {len(e) for e in embeddings}
        if len(dims) > 1:
            # Find the most common dimension
            most_common_dim = max(
                dims, key=lambda d: sum(len(e) == d for e in embeddings)
            )

            # Filter out embeddings with incorrect dimensions
            filtered_vertices = [
                (concept, data)
                for concept, data in vertices
                if len(data["embedding"]) == most_common_dim
            ]
            excluded_vertices.update(
                concept
                for concept, data in vertices
                if len(data["embedding"]) != most_common_dim
            )

            # Update embeddings and vertices for this group
            embeddings = [data["embedding"] for _, data in filtered_vertices]
            vertices = filtered_vertices

        if len(embeddings) > 1:  # PCA requires at least two samples
            embeddings = np.array(embeddings)  # Ensure embeddings are a NumPy array
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(embeddings)
        else:
            # If only one vertex in the group, set arbitrary coordinates
            reduced = [[0, 0]] * len(vertices)

        # Store reduced embeddings for this group
        for i, (concept, _) in enumerate(vertices):
            reduced_embeddings[concept] = reduced[i]

    print(
        f"Excluded vertices due to inconsistent embedding dimensions: {excluded_vertices}"
    )

    # Extract vertex coordinates
    x_nodes, y_nodes, z_nodes = [], [], []
    node_labels = []
    colours = []

    for concept, data in graph.get_all_vertices():
        if concept in excluded_vertices:
            continue  # Skip excluded vertices
        z = z_coordinate(data)
        reduced = reduced_embeddings[concept]
        x_nodes.append(reduced[0])  # PCA component 1
        y_nodes.append(reduced[1])  # PCA component 2
        z_nodes.append(z)
        node_labels.append(concept)
        colours.append(colour(data))

    # Extract edge coordinates and labels
    edge_x, edge_y, edge_z = [], [], []
    edge_labels, edge_label_x, edge_label_y, edge_label_z = [], [], [], []
    for from_label, to_label, key, edge_data in graph.get_all_edges():
        if from_label in excluded_vertices or to_label in excluded_vertices:
            continue  # Skip edges involving excluded vertices
        reduced_v1 = reduced_embeddings[from_label]
        reduced_v2 = reduced_embeddings[to_label]

        edge_x.extend([reduced_v1[0], reduced_v2[0], None])
        edge_y.extend([reduced_v1[1], reduced_v2[1], None])
        edge_z.extend([z_coordinate(graph.graph.nodes[from_label]), z_coordinate(graph.graph.nodes[to_label]), None])

        # Add label in the middle of the edge
        edge_label_x.append((reduced_v1[0] + reduced_v2[0]) / 2)
        edge_label_y.append((reduced_v1[1] + reduced_v2[1]) / 2)
        edge_label_z.append((z_coordinate(graph.graph.nodes[from_label]) + z_coordinate(graph.graph.nodes[to_label])) / 2)
        edge_labels.append(edge_data["label"])

    # Create the edge traces
    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(width=2, color="gray"),
        name="Edges",
    )

    # Create the node traces
    node_trace = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode="markers+text",
        marker=dict(size=8, color=colours, opacity=0.8),
        text=node_labels,
        textposition="top center",
        name="Nodes",
    )

    # Create edge label trace
    edge_label_trace = go.Scatter3d(
        x=edge_label_x,
        y=edge_label_y,
        z=edge_label_z,
        mode="text",
        text=edge_labels,
        textposition="middle center",
        name="Edge Labels",
    )

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace])
    fig.update_layout(
        title="3D Graph Visualization",
        scene=dict(
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            zaxis_title="Z Axis",
        ),
        showlegend=False,
    )

    fig.update_layout(
        width=1000,  # Width in pixels
        height=1000,  # Height in pixels
    )
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )

    fig.show()
