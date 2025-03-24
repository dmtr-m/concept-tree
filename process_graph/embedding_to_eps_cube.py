import sys
import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from typing import List, Tuple
from numpy.typing import NDArray

from directed_graph.vertex import Vertex


def contract_to_cube(epsilon: float, vertex: Vertex) -> NDArray:
    return vertex.embedding // epsilon


def contract_all_vertices(epsilon: float, vertices: List[Vertex]) -> List[List[Tuple[NDArray, Vertex]]]:
    words_to_cube = list()
    bigram_to_cube = list()
    trigram_to_cube = list()

    for vertex in vertices:
        if len(vertex.words_of_concept) == 1:
            words_to_cube.append((contract_to_cube(epsilon, vertex), vertex.concept))
        if len(vertex.words_of_concept) == 2:
            bigram_to_cube.append((contract_to_cube(epsilon, vertex), vertex.concept))
        if len(vertex.words_of_concept) == 3:
            trigram_to_cube.append((contract_to_cube(epsilon, vertex), vertex.concept))

    return [words_to_cube, bigram_to_cube, trigram_to_cube]


def split_graph_into_cubes(epsilon: float, vertices: List[Vertex]):
    vertices_to_cubes = contract_all_vertices(epsilon, vertices)
    for i in range(len(vertices_to_cubes)):
        vertices_to_cubes[i].sort(key=lambda x: x[0].tolist())
    
    return vertices_to_cubes

def get_vertices_by_cubes(vertices_to_cubes: List[List[Tuple[NDArray, Vertex]]]) -> List[List[str]]:
    grouped_vertices = [
        None if len(vertices_to_cubes[0]) == 0 else [set([vertices_to_cubes[0][0][1]])],
        None if len(vertices_to_cubes[1]) == 0 else [set([vertices_to_cubes[1][0][1]])],
        None if len(vertices_to_cubes[2]) == 0 else [set([vertices_to_cubes[2][0][1]])],
    ]

    for dim in range(len(vertices_to_cubes)): # iterate through [word, bigram, trigram]
        group_id = 0

        for i in range(1, len(vertices_to_cubes[dim])):
            if (vertices_to_cubes[dim][i][0] == vertices_to_cubes[dim][i - 1][0]).all():
                grouped_vertices[dim][group_id].add(vertices_to_cubes[dim][i][1])
            else:
                group_id += 1
                grouped_vertices[dim].append(set([vertices_to_cubes[dim][i][1]]))
    
    return grouped_vertices