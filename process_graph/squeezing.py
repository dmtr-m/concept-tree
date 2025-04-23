"""This module provides functionality to squeeze embeddings into clusters based on similarity."""

import sys
import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from directed_graph.vertex import Vertex

import numpy as np

def squeeze(eps: tuple[float], vertices: list[Vertex]) -> list[tuple[dict[str, str], dict[str, list[str]]]]:
    words = dict()
    bigrams = dict()
    trigrams = dict()

    for vertex in vertices:
        if len(vertex.words_of_concept) == 1:
            words[vertex.concept] = vertex.embedding
        if len(vertex.words_of_concept) == 2:
            bigrams[vertex.concept] = vertex.embedding
        if len(vertex.words_of_concept) == 3:
            trigrams[vertex.concept] = vertex.embedding
    
    words_word_to_cluster, words_merged_words = squeeze_by_dict(words, eps[0])
    bigrams_word_to_cluster, bigrams_merged_words = squeeze_by_dict(bigrams, eps[1])
    trigrams_word_to_cluster, trigrams_merged_words = squeeze_by_dict(trigrams, eps[2])

    return [
        (words_word_to_cluster, words_merged_words),
        (bigrams_word_to_cluster, bigrams_merged_words),
        (trigrams_word_to_cluster, trigrams_merged_words),
    ]

def squeeze_by_dict(embeddings: dict[str, np.typing.ArrayLike], similarity_threshold: float) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Squeeze embeddings into clusters based on a similarity threshold.

    Parameters:
        embeddings (dict[str, np.typing.ArrayLike]): A dictionary where keys are words and values are their corresponding embeddings.

    Returns:
        dict[str, str]: A dictionary mapping each word to its corresponding cluster representative.
        dict[str, list[str]]: A dictionary mapping each cluster representative to a list of words in that cluster.
    """
    merged_embeddings = {}
    word_to_cluster = {}
    visited = set()

    sorted_words = sorted(embeddings.keys(), reverse=True)

    for word1 in sorted_words:
        if word1 in visited:
            continue  # Skip if the word has already been visited

        visited.add(word1)
        merged_embeddings[word1] = [word1]
        word_to_cluster[word1] = word1
        
        for word2 in sorted_words:
            if word2 in visited:
                continue  # Skip if the word has already been visited
            
            vec1 = embeddings[word1]
            vec2 = embeddings[word2]

            try:
                distance = np.linalg.norm(vec1 - vec2)  # Calculate the distance between the two embeddings
            except:
                print(f"Failed to calculate distance for: {(word1, len(vec1))}, {(word2, len(vec2))}")
            
            if distance < similarity_threshold:
                visited.add(word2)
                word_to_cluster[word2] = word1  # Assign word2 to the cluster of word1
                merged_embeddings[word1] += [word2]  # Merge word2 into the cluster of word1

    return word_to_cluster, merged_embeddings
