from typing import Protocol

import numpy as np


class Similarity(Protocol):
    def compute_similarity(self, embedding1, embedding2) -> float:
        """Compute similarity of vectors"""


class SimilarityEuclidean(Similarity):
    def compute_similarity(self, embedding1, embedding2) -> float:
        return np.sqrt(
            np.sum((embedding1 - embedding2) ** 2)
        )