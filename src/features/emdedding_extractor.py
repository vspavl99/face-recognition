from typing import Protocol, List

from insightface.app import FaceAnalysis


class EmbeddingModel(Protocol):
    def get_embeddings(self, image) -> List:
        """Return embeddings of all faces in image"""
