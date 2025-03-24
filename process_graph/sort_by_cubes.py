import sys
import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from directed_graph.vertex import Vertex

from typing import List, Tuple
from numpy.typing import NDArray

def sort_by_cubes(vertex_cube: List[Tuple[Vertex, NDArray]]) -> List[Tuple[Vertex, NDArray]]:
    vertex_cube.sort(key=lambda x: x[0].tolist())
    return vertex_cube
