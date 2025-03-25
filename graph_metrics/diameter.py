import sys
import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
from directed_graph.graph import Graph


def diameter(graph: Graph) -> tuple[float]:
    dimensions = 0
    for vertice in graph.vertices.values():
        dimensions = max(dimensions, len(vertice.words_of_concept))

    diameter = [0] * dimensions
    for vertice1 in graph.vertices.values():
        for vertice2 in graph.vertices.values():
            if len(vertice1.words_of_concept) == len(vertice2.words_of_concept):
                ind = len(vertice1.words_of_concept) - 1
                diameter[ind] = max(diameter[ind], np.linalg.norm(vertice1.embedding - vertice2.embedding))
    return tuple(diameter)
