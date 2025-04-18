import numpy as np
import random
from numpy.typing import NDArray
import spacy

embeddings_dict = np.load(
    "embeddings/english_newnewsentencelit_cbow_100_min_df=3_dict.npy", allow_pickle=True
).item()
embeddings_shape = None
for value in embeddings_dict.values():
    embeddings_shape = value.shape
    break

# Load the English NLP model from spaCy
nlp = spacy.load("en_core_web_sm")


def get_word_importance(token):
    """
    Define importance of a word based on its part of speech (POS).
    """
    importance = {
        "NOUN": 3,  # Highest priority (objects, main concepts)
        "PROPN": 3,  # Proper nouns also important
        "VERB": 2,  # Actions are secondary
        "ADJ": 1,  # Adjectives describe nouns, lower priority
        "ADV": 1,  # Adverbs also lower priority
        "ADP": 0,  # Prepositions (of, in, at, etc.)
        "DET": 0,  # Determiners (the, a, an, etc.)
        "PRON": 0,  # Pronouns (he, she, it)
        "CONJ": 0,  # Conjunctions (and, but, or)
        "PART": 0,  # Particles (to, not)
    }
    return importance.get(token.pos_, 0)  # Default to 0 if unknown


def reorder_ngram(phrase):
    """
    Reorder words in a phrase based on importance and dependencies.
    """
    doc = nlp(phrase)

    # Assign importance to each token
    words_with_importance = [
        (token.text, get_word_importance(token), token.dep_) for token in doc
    ]

    # Sort by importance first (descending), then by original dependency order
    sorted_words = sorted(words_with_importance, key=lambda x: (-x[1], x[2]))

    # Return reordered phrase
    return [word[0] for word in sorted_words]


def get_embedding(words: list[str]) -> NDArray[np.float64]:
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
        if len(words) == 1:
            return (
                embeddings_dict[words[0]]
                if words[0] in embeddings_dict
                else np.zeros(embeddings_shape)
            )
        else:
            sorted_words = reorder_ngram(" ".join(words))
            embeddings = [
                (
                    embeddings_dict[word]
                    # if word in embeddings_dict
                    # else np.zeros((embeddings_shape))
                )
                for word in sorted_words
            ]
            return np.concatenate(embeddings)  # Final concatenated embedding
    except Exception as e:
        raise ValueError(f"Failed to generate embedding for words: {words}: {e}") from e
        # return np.random.rand(3) / 10
