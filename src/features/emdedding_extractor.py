from typing import Protocol, List

from insightface.app import FaceAnalysis


class EmbeddingModel(Protocol):
    def get_embeddings(self, image) -> List:
        """Return embeddings of all faces in image"""


class EmbeddingModelInsightface(EmbeddingModel):
    def __init__(self):
        """
        Class for extraction embeddings of faces via insightface
        """

        self._app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
        self._app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embeddings(self, image) -> List:
        predictions = self._app.get(image)
        return [prediction.embeddings for prediction in predictions]
