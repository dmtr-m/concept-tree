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


def process_all_vertices(epsilon: float, vertices: List[Vertex]) -> List[Tuple[NDArray, Vertex]]:
    vertice_to_cube_map = list()
    for vertex in vertices:
        vertice_to_cube_map.append((contract_to_cube(epsilon, vertex), vertex.concept))

    return vertice_to_cube_map
