import unittest

import sys
import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from directed_graph.graph import Graph

from concurrent_fuzzy_rdf_match import FuzzyRDFMatcher

class TestFuzzyRDFMatcher(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()

        self.graph.add_vertex("A")
        self.graph.add_vertex("B")
        self.graph.add_vertex("C")
        self.graph.add_edge("A", "B", "connects")
        self.graph.add_edge("B", "C", "links")

        self.matcher = FuzzyRDFMatcher(self.graph, precision=0.95)
    
    def test_exact_match(self):
        pattern = Graph()

        pattern.add_vertex("A")
        pattern.add_vertex("B")
        pattern.add_edge("A", "B", "connects")

        matches = self.matcher.match(pattern)
        self.assertEqual(len(matches), 1)
    
    def test_no_match(self):
        pattern = Graph()

        pattern.add_vertex("X")
        pattern.add_vertex("Y")
        pattern.add_edge("X", "Y", "unknown")

        matches = self.matcher.match(pattern)
        self.assertEqual(len(matches), 0)
    
    def test_partial_match(self):
        pattern = Graph()

        pattern.add_vertex("A")
        pattern.add_vertex("B")
        pattern.add_vertex("D")
        pattern.add_edge("A", "B", "connects")
        pattern.add_edge("B", "D", "unknown")

        matches = self.matcher.match(pattern)
        self.assertEqual(len(matches), 0)

if __name__ == "__main__":
    unittest.main()
