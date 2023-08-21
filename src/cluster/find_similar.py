import numpy as np

from src.cluster.similarity import Similarity

class FindSimilar:
    def __init__(self, anchor: np.ndarray, similarity: Similarity):
        self._anchor = anchor
        self._neighbors = []
        self._similarity = similarity

    def check_neighborhoods(self, embedding):
        similarity = self._similarity.compute_similarity(self._anchor, embedding)
        return similarity