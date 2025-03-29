import sys
import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from typing import List, Dict, Tuple
from directed_graph.graph import Edge

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import normalize

from itertools import product
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import plotly.graph_objects as go


from sklearn.preprocessing import normalize


def cluster_and_evaluate_all_sizes(
    edges: List[Dict],
    params_by_size: Dict[int, Dict[str, float]],
    standardize: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], Dict[int, Dict[str, str]]]:
    """
    Кластеризует ребра с разными размерами эмбеддингов и вычисляет метрики качества кластеризации.
    Args:
        edges: Список объектов Edge, каждый из которых содержит поле `embedding`.
        params_by_size: Словарь параметров для каждой размерности эмбеддингов.
                        Пример: {300: {"eps": 0.5, "min_samples": 2}, 768: {"eps": 0.3, "min_samples": 5}}
        standardize: Флаг для нормализации эмбеддингов по евклидовой норме.
    Returns:
        Кортеж из двух элементов:
            1. Словарь метрик качества кластеризации для каждой группы размерностей.
            2. Вложенный словарь матчинга:
               - Первый уровень: размер эмбеддингов (int).
               - Второй уровень: словарь, где ключ — это уникальный идентификатор ребра,
                 а значение — метка центрального ребра для данного кластера.
    """
    # Группировка ребер по размеру эмбеддинга
    embedding_size_groups = defaultdict(list)
    for edge in edges:
        embedding_size = len(edge.embedding)
        embedding_size_groups[embedding_size].append(edge)

    all_metrics = {}
    all_matching = {}

    # Обработка каждой группы ребер с одинаковым размером эмбеддинга
    for embedding_size, group in embedding_size_groups.items():
        embeddings = np.array([edge.embedding for edge in group])
        edge_ids = [edge.label for edge in group]  # Идентификаторы ребер

        # Нормализация эмбеддингов, если флаг standardize=True
        if standardize:
            embeddings = normalize(embeddings, norm="l2", axis=1)

        # Параметры для текущей размерности
        if embedding_size not in params_by_size:
            print(
                f"Warning: No parameters provided for embedding size {embedding_size}. Skipping."
            )
            continue
        params = params_by_size[embedding_size]
        eps = params.get("eps", 0.5)
        min_samples = params.get("min_samples", 2)

        # Кластеризация и вычисление метрик
        metrics, matching = cluster_and_evaluate_with_matching(
            embeddings, edge_ids, eps=eps, min_samples=min_samples
        )

        # Сохранение результатов
        all_metrics[f"embedding_size_{embedding_size}"] = metrics
        all_matching[embedding_size] = matching

    return all_metrics, all_matching


def cluster_and_evaluate_with_matching(
    embeddings: np.ndarray,
    edge_labels: List[str],
    eps: float = 0.5,
    min_samples: int = 2,
) -> Tuple[Dict[str, float], np.ndarray, Dict[str, str]]:
    """
    Кластеризует данные с использованием DBSCAN (евклидова метрика),
    вычисляет метрики качества кластеризации и строит матчинг ребер на центральные ребра.

    Args:
        embeddings: Массив эмбеддингов (размерность: [n_samples, n_features]).
        edge_labels: Список идентификаторов ребер, соответствующих эмбеддингам.
        eps: Максимальное расстояние между точками одного кластера.
        min_samples: Минимальное количество точек для формирования кластера.

    Returns:
        Кортеж из трех элементов:
            1. Словарь метрик качества кластеризации.
            2. Массив меток кластеров.
            3. Словарь матчинга ребер на центральные ребра:
               - Ключ: идентификатор ребра.
               - Значение: идентификатор центрального ребра для данного кластера.
    """
    # Кластеризация с использованием DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = dbscan.fit_predict(embeddings)

    # Инициализация словаря метрик
    metrics = {}

    # Вычисление метрик
    unique_labels = sorted(set(labels) - {-1})  # Исключаем шумовые точки (-1)

    if len(unique_labels) > 1:  # Метрики требуют минимум 2 кластера
        try:
            # Silhouette Score
            metrics["Silhouette Score"] = silhouette_score(
                embeddings, labels, metric="cosine"
            )
        except ValueError:
            metrics["Silhouette Score"] = None  # Если не хватает данных для вычисления

        # Davies-Bouldin Index
        metrics["Davies-Bouldin Index"] = davies_bouldin_score(embeddings, labels)

        # Calinski-Harabasz Index
        metrics["Calinski-Harabasz Index"] = calinski_harabasz_score(embeddings, labels)

        # Dunn Index
        def dunn_index(embeddings, labels):
            intra_cluster_dists = []
            inter_cluster_dists = []
            for label in unique_labels:
                cluster_points = embeddings[labels == label]
                if len(cluster_points) > 1:
                    dists = euclidean_distances(cluster_points).max()
                    intra_cluster_dists.append(dists)
            for i, label1 in enumerate(unique_labels):
                for label2 in unique_labels[i + 1 :]:
                    cluster1 = embeddings[labels == label1]
                    cluster2 = embeddings[labels == label2]
                    dists = euclidean_distances(cluster1, cluster2).min()
                    inter_cluster_dists.append(dists)
            if not intra_cluster_dists or not inter_cluster_dists:
                return None
            max_intra = max(intra_cluster_dists)
            min_inter = min(inter_cluster_dists)
            return min_inter / max_intra

        metrics["Dunn Index"] = dunn_index(embeddings, labels)

        # Connectivity Score
        def connectivity_score(embeddings, labels, n_neighbors=10):
            dist_matrix = euclidean_distances(embeddings)
            sorted_indices = np.argsort(dist_matrix, axis=1)[:, 1 : n_neighbors + 1]
            score = 0
            for i, neighbors in enumerate(sorted_indices):
                for j in neighbors:
                    if labels[i] != labels[j]:
                        score += 1
            return score / (len(embeddings) * n_neighbors)

        metrics["Connectivity Score"] = connectivity_score(embeddings, labels)

        # Intra-Cluster Variance
        intra_variance = []
        for label in unique_labels:
            cluster_points = embeddings[labels == label]
            if len(cluster_points) > 0:
                center = cluster_points.mean(axis=0)
                variance = ((cluster_points - center) ** 2).sum(axis=1).mean()
                intra_variance.append(variance)
        metrics["Intra-Cluster Variance"] = (
            np.mean(intra_variance) if intra_variance else None
        )

    else:
        metrics["Silhouette Score"] = None
        metrics["Davies-Bouldin Index"] = None
        metrics["Calinski-Harabasz Index"] = None
        metrics["Dunn Index"] = None
        metrics["Connectivity Score"] = None
        metrics["Intra-Cluster Variance"] = None

    # Cluster Size Distribution
    cluster_sizes = Counter(labels)
    metrics["Cluster Sizes"] = dict(cluster_sizes)

    # Matching edges to central edges in each cluster
    matching = {}
    for label in unique_labels:
        cluster_points = embeddings[labels == label]
        cluster_ids = np.array(edge_labels)[labels == label]

        if len(cluster_points) > 0:
            # Найдем центральное ребро как ближайшее к центроиду кластера
            centroid = cluster_points.mean(axis=0)
            distances_to_centroid = euclidean_distances(
                cluster_points, [centroid]
            ).flatten()
            central_edge_index = np.argmin(distances_to_centroid)
            central_edge_id = cluster_ids[central_edge_index]

            # Создаем матчинг для всех ребер в кластере
            for edge_id in cluster_ids:
                matching[edge_id] = central_edge_id

    # Noise points are matched to themselves
    noise_ids = np.array(edge_labels)[labels == -1]
    for edge_id in noise_ids:
        matching[edge_id] = edge_id

    return metrics, matching


def grid_search_cluster_params(
    edges: List[Dict],
    embedding_sizes: List[int],
    eps_values: List[float] = None,
    min_samples_values: List[int] = None,
    metric_weights: Dict[str, float] = None,
    optimal_cluster_range: Tuple[int, int] = (3, 10),
    standardize: bool = False,  # Новый параметр
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """
    Выполняет Grid Search для выбора оптимальных параметров кластеризации
    с учетом нескольких метрик, включая штраф за количество кластеров.
    Args:
        edges: Список объектов Edge, каждый из которых содержит поле `embedding`.
        embedding_sizes: Список размерностей эмбеддингов.
        eps_values: Список значений eps для перебора (если None, определяется автоматически).
        min_samples_values: Список значений min_samples для перебора.
        metric_weights: Веса для каждой метрики (по умолчанию все метрики равнозначны).
        optimal_cluster_range: Диапазон оптимального числа кластеров (min_clusters, max_clusters).
        standardize: Флаг для нормализации эмбеддингов по евклидовой норме.
    Returns:
        Кортеж из двух элементов:
            1. Словарь оптимальных параметров для каждой размерности.
            2. Словарь исходных метрик для каждой размерности.
    """
    if min_samples_values is None:
        min_samples_values = [2, 3, 5]
    if metric_weights is None:
        metric_weights = {
            "Silhouette Score": 2.0,
            "Davies-Bouldin Index": -1.0,  # Минимизация
            "Calinski-Harabasz Index": 1.0,
            "Dunn Index": 1.0,
            "Connectivity Score": -1.0,  # Минимизация
            "Intra-Cluster Variance": -1.0,  # Минимизация
            "Noise Ratio": -3.0,  # Минимизация
            "Cluster Count Penalty": -2.0,  # Штраф за количество кластеров
        }

    # Группировка ребер по размеру эмбеддинга
    embedding_size_groups = defaultdict(list)
    for edge in edges:
        embedding_size = len(edge.embedding)
        embedding_size_groups[embedding_size].append(edge)

    best_params_by_size = {}
    best_metrics_by_size = {}

    # Обработка каждой группы ребер с одинаковым размером эмбеддинга
    for embedding_size in embedding_sizes:
        if embedding_size not in embedding_size_groups:
            print(
                f"Warning: No data found for embedding size {embedding_size}. Skipping."
            )
            continue

        embeddings = np.array(
            [edge.embedding for edge in embedding_size_groups[embedding_size]]
        )

        # Нормализация эмбеддингов, если флаг standardize=True
        if standardize:
            embeddings = normalize(embeddings, norm="l2", axis=1)

        # Автоматическое определение диапазона eps, если он не задан
        if eps_values is None:
            distances = euclidean_distances(embeddings)
            distances = distances[np.triu_indices_from(distances, k=1)]
            median_distance = np.median(distances)
            percentile_90 = np.percentile(distances, 90)
            eps_values = np.linspace(median_distance, percentile_90, 5)

        # Генерация всех комбинаций параметров
        param_combinations = list(product(eps_values, min_samples_values))
        best_aggregated_score = -np.inf
        best_params = None
        best_metrics = None

        # Перебор всех комбинаций параметров
        for eps, min_samples in param_combinations:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
            labels = dbscan.fit_predict(embeddings)

            # Вычисление метрик
            metrics = {}
            unique_labels = sorted(set(labels) - {-1})  # Исключаем шумовые точки (-1)
            if len(unique_labels) > 1:
                try:
                    metrics["Silhouette Score"] = silhouette_score(
                        embeddings, labels, metric="cosine"
                    )
                except ValueError:
                    metrics["Silhouette Score"] = -np.inf
                metrics["Davies-Bouldin Index"] = davies_bouldin_score(
                    embeddings, labels
                )
                metrics["Calinski-Harabasz Index"] = calinski_harabasz_score(
                    embeddings, labels
                )

                # Dunn Index
                def dunn_index(embeddings, labels):
                    intra_cluster_dists = []
                    inter_cluster_dists = []
                    for label in unique_labels:
                        cluster_points = embeddings[labels == label]
                        if len(cluster_points) > 1:
                            dists = euclidean_distances(cluster_points).max()
                            intra_cluster_dists.append(dists)
                    for i, label1 in enumerate(unique_labels):
                        for label2 in unique_labels[i + 1 :]:
                            cluster1 = embeddings[labels == label1]
                            cluster2 = embeddings[labels == label2]
                            dists = euclidean_distances(cluster1, cluster2).min()
                            inter_cluster_dists.append(dists)
                    if not intra_cluster_dists or not inter_cluster_dists:
                        return None
                    max_intra = max(intra_cluster_dists)
                    min_inter = min(inter_cluster_dists)
                    return min_inter / max_intra

                metrics["Dunn Index"] = dunn_index(embeddings, labels)

                # Connectivity Score
                def connectivity_score(embeddings, labels, n_neighbors=10):
                    dist_matrix = euclidean_distances(embeddings)
                    sorted_indices = np.argsort(dist_matrix, axis=1)[
                        :, 1 : n_neighbors + 1
                    ]
                    score = 0
                    for i, neighbors in enumerate(sorted_indices):
                        for j in neighbors:
                            if labels[i] != labels[j]:
                                score += 1
                    return score / (len(embeddings) * n_neighbors)

                metrics["Connectivity Score"] = connectivity_score(embeddings, labels)
                # Intra-Cluster Variance
                intra_variance = []
                for label in unique_labels:
                    cluster_points = embeddings[labels == label]
                    if len(cluster_points) > 0:
                        center = cluster_points.mean(axis=0)
                        variance = ((cluster_points - center) ** 2).sum(axis=1).mean()
                        intra_variance.append(variance)
                metrics["Intra-Cluster Variance"] = (
                    np.mean(intra_variance) if intra_variance else None
                )
                # Noise Ratio
                noise_ratio = sum(1 for label in labels if label == -1) / len(labels)
                metrics["Noise Ratio"] = noise_ratio
                # Cluster Count Penalty
                num_clusters = len(unique_labels)
                min_clusters, max_clusters = optimal_cluster_range
                if num_clusters < min_clusters:
                    penalty = (
                        min_clusters - num_clusters
                    ) ** 2  # Штраф за слишком мало кластеров
                elif num_clusters > max_clusters:
                    penalty = (
                        num_clusters - max_clusters
                    ) ** 2  # Штраф за слишком много кластеров
                else:
                    penalty = 0  # Нет штрафа в оптимальном диапазоне
                metrics["Cluster Count Penalty"] = penalty
            else:
                metrics = {
                    "Silhouette Score": -np.inf,
                    "Davies-Bouldin Index": np.inf,
                    "Calinski-Harabasz Index": -np.inf,
                    "Dunn Index": -np.inf,
                    "Connectivity Score": np.inf,
                    "Intra-Cluster Variance": np.inf,
                    "Noise Ratio": 1.0,  # Все точки — шум
                    "Cluster Count Penalty": np.inf,  # Максимальный штраф
                }

            # Проверка на наличие бесконечных значений
            if any(value is None or np.isinf(value) for value in metrics.values()):
                print(
                    f"Embedding size {embedding_size}, eps={eps:.4f}, min_samples={min_samples}: "
                    "Skipping due to infinite or undefined metric values."
                )
                continue

            # Нормализация метрик и вычисление агрегированной метрики
            aggregated_score = 0.0
            for metric_name, weight in metric_weights.items():
                value = metrics[metric_name]
                if metric_name in [
                    "Davies-Bouldin Index",
                    "Connectivity Score",
                    "Intra-Cluster Variance",
                    "Noise Ratio",
                    "Cluster Count Penalty",
                ]:
                    normalized_value = 1 / (1 + value + 0.001)  # Минимизация
                else:
                    normalized_value = value / (1 + value + 0.001)  # Максимизация
                aggregated_score += weight * normalized_value

            # Обновление лучших параметров
            if aggregated_score > best_aggregated_score:
                best_aggregated_score = aggregated_score
                best_params = {"eps": eps, "min_samples": min_samples}
                best_metrics = metrics

        # Сохранение результатов
        if best_params is not None:
            best_params_by_size[embedding_size] = best_params
            best_metrics_by_size[embedding_size] = best_metrics
            print(f"Embedding size {embedding_size}: Best params = {best_params}")
            print("Best Metrics:")
            for metric_name, value in best_metrics.items():
                print(f"  {metric_name}: {value}")
        else:
            print(
                f"Embedding size {embedding_size}: No valid parameter combinations found."
            )

    return best_params_by_size, best_metrics_by_size


def analyze_distance_distributions(edges: list) -> Dict[int, Dict[str, float]]:
    """
    Анализирует распределение попарных расстояний для эмбеддингов с разными размерностями.

    Args:
        edges: Список объектов Edge, каждый из которых содержит поле `embedding`.

    Returns:
        Словарь, где ключ — это размерность эмбеддинга, а значение — словарь с метриками:
            - "median_distance": Медиана расстояний.
            - "percentile_90": 90-й процентиль расстояний.
    """
    # Группировка ребер по размеру эмбеддинга
    embedding_size_groups = defaultdict(list)
    for edge in edges:
        embedding_size = len(edge.embedding)
        embedding_size_groups[embedding_size].append(edge)

    distance_stats_by_size = {}

    # Обработка каждой группы ребер с одинаковым размером эмбеддинга
    for embedding_size, group in embedding_size_groups.items():
        embeddings = np.array([edge.embedding for edge in group])

        # Вычисление попарных евклидовых расстояний
        distances = cosine_distances(embeddings)
        distances = distances[
            np.triu_indices_from(distances, k=1)
        ]  # Верхний треугольник матрицы без диагонали

        # Построение гистограммы распределения расстояний
        plt.hist(distances, bins=50, alpha=0.7, label=f"Size {embedding_size}")
        plt.title(
            f"Distribution of pairwise distances (Embedding size {embedding_size})"
        )
        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

        # Вычисление медианы и 90-го процентиля
        median_distance = np.median(distances)
        percentile_90 = np.percentile(distances, 90)

        # Сохранение результатов
        distance_stats_by_size[embedding_size] = {
            "median_distance": median_distance,
            "percentile_90": percentile_90,
        }

        print(
            f"Embedding size {embedding_size}: Median distance = {median_distance:.4f}, "
            f"90th percentile = {percentile_90:.4f}"
        )

    return distance_stats_by_size


def plot_clusters_with_pca(edges: list, matching: dict) -> None:
    """
    Строит интерактивные графики кластеров для каждого размера эмбеддингов.
    Использует PCA для уменьшения размерности до 2D.

    Args:
        edges: Список объектов Edge, каждый из которых содержит поля `.label`, `.embedding`.
        matching: Вложенный словарь матчинга:
                  - Первый уровень: размер эмбеддингов (int).
                  - Второй уровень: словарь, где ключ — это метка ребра (поле .label),
                    а значение — метка центрального ребра для данного кластера.
    """
    # Группировка ребер по размеру эмбеддинга
    embedding_size_groups = defaultdict(list)
    for edge in edges:
        embedding_size = len(edge.embedding)
        embedding_size_groups[embedding_size].append(edge)

    # Обработка каждой группы ребер с одинаковым размером эмбеддинга
    for embedding_size, group in embedding_size_groups.items():
        if embedding_size not in matching:
            print(
                f"Warning: No matching provided for embedding size {embedding_size}. Skipping."
            )
            continue

        # Извлечение данных
        embeddings = np.array([edge.embedding for edge in group])
        labels = [edge.label for edge in group]  # Метки ребер

        # Применение PCA для уменьшения размерности до 2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        # Получение матчинга для текущего размера эмбеддингов
        size_matching = matching[embedding_size]

        # Создание цветовой палитры для кластеров
        unique_central_labels = set(size_matching.values())
        colors = [
            f"rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})"
            for _ in range(len(unique_central_labels))
        ]
        color_map = dict(zip(unique_central_labels, colors))

        # Подготовка данных для графика
        scatter_data = []
        for central_label in unique_central_labels:
            cluster_indices = [
                i
                for i, label in enumerate(labels)
                if size_matching[label] == central_label
            ]
            if not cluster_indices:
                continue

            # Координаты точек кластера
            cluster_coords = embeddings_2d[cluster_indices]
            cluster_labels = [labels[i] for i in cluster_indices]

            # Главное слово кластера (из словаря matching)
            main_word = central_label

            # Добавление данных для кластера
            scatter_data.append(
                go.Scatter(
                    x=cluster_coords[:, 0],
                    y=cluster_coords[:, 1],
                    mode="markers+text",
                    name=f"Cluster ({main_word})",
                    text=cluster_labels,  # Текст для каждой точки
                    marker=dict(color=color_map[central_label]),
                    textposition="top center",
                )
            )

        # Создание графика
        fig = go.Figure(data=scatter_data)
        fig.update_layout(
            title=f"Clusters for Embedding Size {embedding_size}",
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            hovermode="closest",
        )

        # Отображение графика
        fig.show()
