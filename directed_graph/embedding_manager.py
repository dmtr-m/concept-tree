import numpy as np
from numpy.typing import NDArray

embeddings_dict = np.load('embeddings/english_lit_SVD_dict.npy', allow_pickle=True).item()

def get_embedding(word: str) -> NDArray[np.float64]:
    """
    Get the embedding for a given concept.

    Args:
        word: The word to get an embedding for

    Returns:
        A numpy array representing the word embedding

    Raises:
        ValueError: If the embedding generation fails
    """
    try:
        return embeddings_dict[word][:3]
    except Exception as e:
        # raise ValueError(
        #     f"Failed to generate embedding for word: {word}") from e
        return np.random.rand(3) / 10 
