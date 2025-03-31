import sys
import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from typing import List, Dict, Tuple, Any
from directed_graph.graph import Edge, Graph

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

from itertools import product
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import plotly.graph_objects as go


from sklearn.preprocessing import normalize

from scipy.sparse import csr_matrix

from tqdm import tqdm


def extract_unique_sorted_embeddings(edges: List[Edge]) -> List[List[float]]:
    """
    Извлекает эмбеддинги с уникальными label и сортирует их лексикографически по label.

    Args:
        edges: Список объектов Edge, каждый из которых содержит поля `embedding` и `label`.

    Returns:
        Список эмбеддингов, отсортированный лексикографически по label.
    """
    # Группировка ребер по label (только уникальные label)
    unique_labels = {}
    for edge in edges:
        if edge.label not in unique_labels:
            unique_labels[edge.label] = edge

    # Сортировка label лексикографически
    sorted_labels = sorted(unique_labels.keys())

    # Извлечение эмбеддингов в порядке отсортированных label
    sorted_embeddings = [unique_labels[label].embedding for label in sorted_labels]

    return sorted_labels, sorted_embeddings


def cluster_and_evaluate_all_sizes(
    embeddings_list: List[np.ndarray],
    labels_list: List[str],
    params_by_size: Dict[int, Dict[str, Any]],
    standardize: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], Dict[int, Dict[str, str]]]:
    """
    Кластеризует эмбеддинги с разными размерами и вычисляет метрики качества кластеризации.
    Args:
        embeddings_list: Список эмбеддингов (np.ndarray).
        labels_list: Список соответствующих лейблов для эмбеддингов.
        params_by_size: Словарь параметров для каждой размерности эмбеддингов.
                        Пример: {300: {"model": "DBSCAN", "params": {"eps": 0.5, "min_samples": 2}},
                                 768: {"model": "AgglomerativeClustering", "params": {"n_clusters": 10}}}
        standardize: Флаг для нормализации эмбеддингов по евклидовой норме.
    Returns:
        Кортеж из двух элементов:
            1. Словарь метрик качества кластеризации для каждой группы размерностей.
            2. Вложенный словарь матчинга:
               - Первый уровень: размер эмбеддингов (int).
               - Второй уровень: словарь, где ключ — это уникальный идентификатор ребра,
                 а значение — метка центрального ребра для данного кластера.
    """
    if len(embeddings_list) != len(labels_list):
        raise ValueError("embeddings_list and labels_list must have the same length.")

    # Группировка эмбеддингов по их размеру
    embedding_size_groups = defaultdict(list)
    for emb, label in zip(embeddings_list, labels_list):
        embedding_size = len(emb)
        embedding_size_groups[embedding_size].append((emb, label))

    all_metrics = {}
    all_matching = {}

    # Обработка каждой группы эмбеддингов с одинаковым размером
    for embedding_size, group in tqdm(embedding_size_groups.items()):
        embeddings = np.array([item[0] for item in group])  # Эмбеддинги
        edge_ids = [item[1] for item in group]  # Идентификаторы

        # Нормализация эмбеддингов, если флаг standardize=True
        if standardize:
            embeddings = normalize(embeddings, norm="l2", axis=1)

        # Параметры для текущей размерности
        if embedding_size not in params_by_size:
            print(
                f"Warning: No parameters provided for embedding size {embedding_size}. Skipping."
            )
            continue
        model_config = params_by_size[embedding_size]
        model_name = model_config["model"]
        model_params = model_config["params"]

        # Создание модели на основе переданных параметров
        try:
            if model_name == "DBSCAN":
                model = DBSCAN(**model_params)
            elif model_name == "AgglomerativeClustering":
                model = AgglomerativeClustering(**model_params)
            else:
                raise ValueError(f"Модель {model_name} не поддерживается.")
        except Exception as e:
            print(f"Ошибка при создании модели {model_name}: {e}")
            continue

        # Кластеризация и вычисление метрик
        metrics, matching = cluster_and_evaluate_with_matching(
            embeddings, edge_ids, model=model
        )

        # Сохранение результатов
        all_metrics[f"embedding_size_{embedding_size}"] = metrics
        all_matching[embedding_size] = matching

    return all_metrics, all_matching


def cluster_and_evaluate_with_matching(
    embeddings: np.ndarray,
    edge_labels: List[str],
    model: Any,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Кластеризует данные с использованием заданной модели,
    вычисляет метрики качества кластеризации и строит матчинг ребер на центральные ребра.

    Args:
        embeddings: Массив эмбеддингов (размерность: [n_samples, n_features]).
        edge_labels: Список идентификаторов, соответствующих эмбеддингам.
        model: Объект модели для кластеризации (например, DBSCAN или AgglomerativeClustering).

    Returns:
        Кортеж из двух элементов:
            1. Словарь метрик качества кластеризации.
            2. Словарь матчинга ребер на центральные ребра:
               - Ключ: идентификатор ребра.
               - Значение: идентификатор центрального ребра для данного кластера.
    """
    # Кластеризация
    labels = model.fit_predict(embeddings)

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
    unique_edges_ebs: List[np.ndarray],
    embedding_sizes: List[int],
    model_param_grid: Dict[str, Dict[str, list]],
    standardize: bool = False,  # Флаг для нормализации эмбеддингов
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """
    Выполняет Grid Search для выбора оптимальных параметров кластеризации
    с учетом нескольких метрик для каждой размерности эмбеддингов.
    Args:
        unique_edges_ebs: Список эмбеддингов ребер.
        embedding_sizes: Список размерностей эмбеддингов.
        model_param_grid: Словарь, где ключи — названия моделей, а значения — словари параметров для перебора.
                          Пример: {"DBSCAN": {"eps": [0.3, 0.5], "min_samples": [2, 5], "metric": ["cosine", "euclidean"]}}
        standardize: Флаг для нормализации эмбеддингов по евклидовой норме.
    Returns:
        Кортеж из двух элементов:
            1. Словарь оптимальных параметров для каждой размерности.
            2. Словарь исходных метрик для каждой размерности.
    """
    # Группировка ребер по размеру эмбеддинга
    embedding_size_groups = defaultdict(list)
    for edge_emb in unique_edges_ebs:
        embedding_size = len(edge_emb)
        embedding_size_groups[embedding_size].append(edge_emb)

    best_params_by_size = {}
    best_metrics_by_size = {}

    # Обработка каждой группы ребер с одинаковым размером эмбеддинга
    for embedding_size in embedding_sizes:
        if embedding_size not in embedding_size_groups:
            print(
                f"Warning: No data found for embedding size {embedding_size}. Skipping."
            )
            continue

        embeddings = np.array(embedding_size_groups[embedding_size])

        # Нормализация эмбеддингов, если флаг standardize=True
        if standardize:
            embeddings = normalize(embeddings, norm="l2", axis=1)

        # Словарь для хранения метрик для всех комбинаций параметров
        all_metrics = []

        # Перебор всех моделей и их параметров
        for model_name, param_grid in tqdm(model_param_grid.items()):
            keys, values = zip(*param_grid.items())
            param_combinations = [dict(zip(keys, v)) for v in product(*values)]

            for params in tqdm(param_combinations):
                try:
                    # Создание модели с текущими параметрами
                    if model_name == "DBSCAN":
                        model = DBSCAN(**params)
                    elif model_name == "AgglomerativeClustering":
                        if "connectivity" in params:
                            connectivity_matrix_dict = params.pop("connectivity")
                            if connectivity_matrix_dict is not None:
                                params["connectivity"] = connectivity_matrix_dict.get(
                                    embedding_size, None
                                )
                        model = AgglomerativeClustering(**params)
                    else:
                        raise ValueError(f"Модель {model_name} не поддерживается.")
                except Exception as e:
                    print(
                        f"Ошибка при создании модели {model_name} с параметрами {params}: {e}"
                    )
                    continue

                try:
                    # Обучение модели
                    labels = model.fit_predict(embeddings)
                except Exception as e:
                    print(f"Ошибка при обучении модели: {e}")
                    continue

                # Вычисление метрик
                metrics = {}
                unique_labels = sorted(
                    set(labels) - {-1}
                )  # Исключаем шумовые точки (-1)

                if len(unique_labels) > 1:
                    try:
                        metrics["Silhouette Score"] = silhouette_score(
                            embeddings, labels, metric="euclidean"
                        )
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
                                    dists = euclidean_distances(
                                        cluster1, cluster2
                                    ).min()
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

                        metrics["Connectivity Score"] = connectivity_score(
                            embeddings, labels
                        )

                        # Intra-Cluster Variance
                        intra_variance = []
                        for label in unique_labels:
                            cluster_points = embeddings[labels == label]
                            if len(cluster_points) > 0:
                                center = cluster_points.mean(axis=0)
                                variance = (
                                    ((cluster_points - center) ** 2).sum(axis=1).mean()
                                )
                                intra_variance.append(variance)
                        metrics["Intra-Cluster Variance"] = (
                            np.mean(intra_variance) if intra_variance else None
                        )

                    except Exception as e:
                        print(f"Metrics eval exception: {e}")
                        metrics = {
                            key: None
                            for key in [
                                "Silhouette Score",
                                "Davies-Bouldin Index",
                                "Calinski-Harabasz Index",
                                "Dunn Index",
                                "Connectivity Score",
                                "Intra-Cluster Variance",
                            ]
                        }
                else:
                    # Если только один кластер или все точки — шум
                    metrics = {
                        "Silhouette Score": -np.inf,
                        "Davies-Bouldin Index": np.inf,
                        "Calinski-Harabasz Index": -np.inf,
                        "Dunn Index": -np.inf,
                        "Connectivity Score": np.inf,
                        "Intra-Cluster Variance": np.inf,
                    }
                    print("*Unable to compute metrics, only 1 cluster*")

                # Добавляем метрики и параметры в общий список
                all_metrics.append(
                    {
                        "params": {"model": model_name, "params": params},
                        "metrics": metrics,
                    }
                )

        # Проверка, есть ли данные для нормализации
        if not all_metrics:
            print(
                f"Embedding size {embedding_size}: No valid parameter combinations found."
            )
            continue

        # Z-нормализация метрик
        metric_names = [
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz Index",
            "Dunn Index",
            "Connectivity Score",
            "Intra-Cluster Variance",
        ]

        # Определяем направления метрик (большие или маленькие значения лучше)
        metric_directions = {
            "Silhouette Score": 1,  # Большие значения лучше
            "Davies-Bouldin Index": -1,  # Маленькие значения лучше
            "Calinski-Harabasz Index": 1,  # Большие значения лучше
            "Dunn Index": 1,  # Большие значения лучше
            "Connectivity Score": -1,  # Маленькие значения лучше
            "Intra-Cluster Variance": -1,  # Маленькие значения лучше
        }

        for metric_name in metric_names:
            metric_values = [
                m["metrics"][metric_name]
                for m in all_metrics
                if m["metrics"][metric_name] is not None
            ]
            if metric_values:
                scaler = StandardScaler()
                normalized_values = scaler.fit_transform(
                    np.array(metric_values).reshape(-1, 1)
                ).flatten()

                # Умножаем нормализованные значения на направление метрики
                direction = metric_directions[metric_name]
                normalized_values *= direction

                # Заменяем исходные значения на нормализованные
                idx = 0
                for m in all_metrics:
                    if m["metrics"][metric_name] is not None:
                        m["metrics"][metric_name] = normalized_values[idx]
                        idx += 1

        # Вычисление среднего арифметического нормализованных значений
        best_aggregated_score = -np.inf
        best_params = None
        best_metrics = None

        for m in all_metrics:
            aggregated_score = np.mean(
                [value for value in m["metrics"].values() if value is not None]
            )
            if aggregated_score > best_aggregated_score:
                best_aggregated_score = aggregated_score
                best_params = m["params"]
                best_metrics = m["metrics"]

        # Сохранение результатов
        if best_params is not None:
            best_params_by_size[embedding_size] = best_params
            best_metrics_by_size[embedding_size] = best_metrics
            print(
                f"Embedding size {embedding_size}: Best model = {best_params['model']}, Best params = {best_params['params']}"
            )
            print("Best Metrics:")
            for metric_name, value in best_metrics.items():
                print(f"  {metric_name}: {value}")
        else:
            print(
                f"Embedding size {embedding_size}: No valid parameter combinations found."
            )

    return best_params_by_size, best_metrics_by_size


def compute_connectivity_matrix(
    edges: List[Edge],
    embedding_sizes: List[int],
    matrix_type: str = "adjacency",
) -> Dict[int, csr_matrix]:
    connectivity_matrices = {}

    for emb_size in embedding_sizes:
        filtered_edges = [edge for edge in edges if len(edge.embedding) == emb_size]
        if not filtered_edges:
            print(f"Warning: No edges found for embedding size {emb_size}. Skipping.")
            continue

        label_to_edges = defaultdict(list)
        for edge in filtered_edges:
            label_to_edges[edge.label].append(edge)

        unique_labels = sorted(list(label_to_edges.keys()))
        n_unique_labels = len(unique_labels)

        data = []
        row_indices = []
        col_indices = []

        if matrix_type == "adjacency":
            # Матрица на основе смежности ребер
            for i, label1 in enumerate(unique_labels):
                for j, label2 in enumerate(unique_labels):
                    edges1 = label_to_edges[label1]
                    edges2 = label_to_edges[label2]

                    # Проверяем смежность хотя бы одного ребра из группы label1 с любым ребром из группы label2
                    for edge1 in edges1:
                        for edge2 in edges2:
                            if set([edge1.agent_1, edge1.agent_2]).intersection(
                                [edge2.agent_1, edge2.agent_2]
                            ):
                                data.append(1)
                                row_indices.append(i)
                                col_indices.append(j)
                                break

        elif matrix_type == "shortest_path":
            from networkx import Graph as NXGraph
            from networkx.algorithms.shortest_paths.dense import floyd_warshall_numpy

            graph = NXGraph()
            for edge in filtered_edges:
                graph.add_edge(edge.agent_1, edge.agent_2)

            # Предварительное вычисление всех кратчайших путей
            node_list = list(graph.nodes)
            distance_matrix = floyd_warshall_numpy(graph, nodelist=node_list)

            for i, label1 in enumerate(unique_labels):
                for j, label2 in enumerate(unique_labels):
                    edges1 = label_to_edges[label1]
                    edges2 = label_to_edges[label2]

                    # Получаем множества узлов для групп label1 и label2
                    nodes1 = {
                        node_list.index(node)
                        for node in {edge.agent_1 for edge in edges1}.union(
                            {edge.agent_2 for edge in edges1}
                        )
                    }
                    nodes2 = {
                        node_list.index(node)
                        for node in {edge.agent_1 for edge in edges2}.union(
                            {edge.agent_2 for edge in edges2}
                        )
                    }

                    # Находим минимальное расстояние между группами
                    min_distance = float("inf")
                    for node1 in nodes1:
                        for node2 in nodes2:
                            if distance_matrix[node1, node2] < min_distance:
                                min_distance = distance_matrix[node1, node2]

                    if min_distance < float("inf"):
                        data.append(min_distance)
                        row_indices.append(i)
                        col_indices.append(j)

        else:
            raise ValueError(f"Неизвестный тип матрицы: {matrix_type}")

        connectivity_matrix = csr_matrix(
            (data, (row_indices, col_indices)), shape=(n_unique_labels, n_unique_labels)
        )
        connectivity_matrices[emb_size] = connectivity_matrix

    return connectivity_matrices


def analyze_distance_distributions(
    edges_embeddings: np.ndarray, metric: str = "cosine"
) -> Dict[int, Dict[str, float]]:
    """
    Анализирует распределение попарных расстояний для эмбеддингов с разными размерностями,
    используя только ребра с уникальными значениями поля `label`.

    Args:
        edges_embeddings: Список уникальных эмбеддингов
        metric: Метрика для вычисления расстояний ("cosine" или "euclidean").

    Returns:
        Словарь, где ключ — это размерность эмбеддинга, а значение — словарь с метриками:
            - "median_distance": Медиана расстояний.
            - "percentile_90": 90-й процентиль расстояний.
    """

    # Группировка ребер по размеру эмбеддинга
    embedding_size_groups = defaultdict(list)
    for edge_emb in edges_embeddings:
        embedding_size = len(edge_emb)
        embedding_size_groups[embedding_size].append(edge_emb)

    distance_stats_by_size = {}

    # Обработка каждой группы ребер с одинаковым размером эмбеддинга
    for embedding_size, emb_group in embedding_size_groups.items():

        # Вычисление попарных расстояний в зависимости от метрики
        if metric == "cosine":
            distances = cosine_distances(emb_group)
        elif metric == "euclidean":
            distances = euclidean_distances(emb_group)
        else:
            raise ValueError(
                f"Unsupported metric: {metric}. Use 'cosine' or 'euclidean'."
            )

        # Берем только верхний треугольник матрицы без диагонали
        distances = distances[np.triu_indices_from(distances, k=1)]

        # Построение гистограммы распределения расстояний
        plt.figure(figsize=(8, 6))
        plt.hist(distances, bins=50, alpha=0.7, color="blue", edgecolor="black")
        plt.title(
            f"Distribution of pairwise {metric} distances (Embedding size {embedding_size})"
        )
        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle="--", alpha=0.5)
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
            f"Embedding size {embedding_size}: Median {metric} distance = {median_distance:.4f}, "
            f"90th percentile = {percentile_90:.4f}"
        )

    return distance_stats_by_size


def plot_clusters_with_pca(
    unique_edges_embeddings: List[np.ndarray],
    unique_edges_labels: List[str],
    matching: Dict[int, Dict[str, str]],
) -> None:
    """
    Строит интерактивные графики кластеров для каждого размера эмбеддингов.
    Использует PCA для уменьшения размерности до 2D.

    Args:
        unique_edges_embeddings: Список уникальных эмбеддингов (np.ndarray).
        unique_edges_labels: Список соответствующих меток ребер (str).
        matching: Вложенный словарь матчинга:
                  - Первый уровень: размер эмbedding'ов (int).
                  - Второй уровень: словарь, где ключ — это метка ребра,
                    а значение — метка центрального ребра для данного кластера.
    """
    # Проверка, что длина списков совпадает
    if len(unique_edges_embeddings) != len(unique_edges_labels):
        raise ValueError(
            "unique_edges_embeddings and unique_edges_labels must have the same length."
        )

    # Группировка эмbedding'ов по их размеру
    embedding_size_groups = defaultdict(list)
    for emb, label in zip(unique_edges_embeddings, unique_edges_labels):
        embedding_size = len(emb)
        embedding_size_groups[embedding_size].append((emb, label))

    # Обработка каждой группы эмbedding'ов с одинаковым размером
    for embedding_size, group in embedding_size_groups.items():
        if embedding_size not in matching:
            print(
                f"Warning: No matching provided for embedding size {embedding_size}. Skipping."
            )
            continue

        # Извлечение данных
        embeddings = np.array([item[0] for item in group])  # Эмbedding'и
        labels = [item[1] for item in group]  # Метки ребер

        # Применение PCA для уменьшения размерности до 2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        # Получение матчинга для текущего размера эмbedding'ов
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

            # Размер кластера
            cluster_size = len(cluster_indices)

            # Главное слово кластера (из словаря matching)
            main_word = central_label

            # Добавление данных для кластера с указанием размера в названии
            scatter_data.append(
                go.Scatter(
                    x=cluster_coords[:, 0],
                    y=cluster_coords[:, 1],
                    mode="markers+text",
                    name=f"Cluster ({main_word}), Size: {cluster_size}",
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
