from abc import abstractmethod

from insightface.app import FaceAnalysis


class EmbeddingModel:
    @abstractmethod
    def get_embeddings(self, image) -> list:
        """Return embeddings of all faces in image"""


class EmbeddingModelInsightface(EmbeddingModel):
    def __init__(self):
        """
        Class for extraction embeddings of faces via insightface
        """

        self._app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
        self._app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embeddings(self, image) -> list:
        predictions = self._app.get(image)

        predictions = self._filter_predictions(predictions)
        return [prediction['embedding'] for prediction in predictions]

    @staticmethod
    def _filter_predictions(predictions) -> list:
        """
        Keep only one face-detection with the largest square
        :param predictions: list of all predictions on image
        :return: face with the largest square
        """

        filtered_predictions = []

        max_square = 0
        for prediction in predictions:
            square = (prediction['bbox'][2] - prediction['bbox'][0]) * (prediction['bbox'][3] - prediction['bbox'][1])

            if square > max_square:
                max_square = square
                filtered_predictions = [prediction]

        return filtered_predictions