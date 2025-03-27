import sys
import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from typing import List, Dict

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from directed_graph.graph import Edge


from collections import defaultdict


def cluster_edges_by_embeddings_dbscan(
    edges: List[Edge], eps: float = 0.5, min_samples: int = 2
) -> Dict[str, str]:
    """
    Кластеризует ребра по их эмбеддингам с использованием DBSCAN и сопоставляет лейблы ребер с лейблами центров кластеров.

    Args:
        edges: Список объектов Edge.
        eps: Максимальное расстояние между точками одного кластера.
        min_samples: Минимальное количество точек для формирования кластера.

    Returns:
        Словарь сопоставления: лейбл_ребра -> лейбл_центра.
    """
    # Группировка ребер по размеру эмбеддинга
    embedding_size_groups = defaultdict(list)
    for edge in edges:
        embedding_size = len(edge.embedding)
        embedding_size_groups[embedding_size].append(edge)

    label_to_cluster_center = {}

    # Обработка каждой группы ребер с одинаковым размером эмбеддинга
    for embedding_size, group in embedding_size_groups.items():
        # Извлечение эмбеддингов
        embeddings = np.array([edge.embedding for edge in group])

        # Кластеризация с использованием DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = dbscan.fit_predict(embeddings)

        # Определение центров кластеров
        unique_labels = set(labels) - {-1}  # Исключаем шумовые точки (-1)
        cluster_centers = {}

        for label_id in unique_labels:
            # Находим все точки, принадлежащие текущему кластеру
            cluster_points = embeddings[labels == label_id]

            # Вычисляем центр кластера как среднее значение всех точек
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers[label_id] = cluster_center

        # Нахождение ближайшего центра для каждого ребра
        for edge, cluster_id in zip(group, labels):
            if cluster_id == -1:  # Пропускаем шумовые точки
                continue

            closest_center = cluster_centers[cluster_id]

            # Нахождение ребра, которое ближе всего к центру кластера
            similarities = cosine_similarity([closest_center], embeddings)[0]
            closest_edge_index = np.argmax(similarities)
            closest_edge = group[closest_edge_index]

            # Сопоставление лейбла ребра с лейблом центра
            label_to_cluster_center[edge.label] = closest_edge.label

    return label_to_cluster_center
