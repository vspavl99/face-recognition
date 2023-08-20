from typing import Iterator

from src.utils.embedding_saver import EmbeddingSaver
from src.models.emdedding_extractor import EmbeddingModel


class FaceEmbeddingGenerator:
    def __init__(self, model: EmbeddingModel, images_iter: Iterator, embeddings_saver: EmbeddingSaver):
        """
        Class for generating embeddings of faces on images
        :param model: model for extracting embeddings
        :param images_iter: paths to images
        :param embeddings_saver: class for saving embeddings of faces
        """

        self._model = model
        self._images_iter = images_iter
        self._embeddings_saver = embeddings_saver

    def generate(self):
        for (image, image_name)  in self._images_iter:
            embeddings_per_image = self._model.get_embeddings(image)
            self._embeddings_saver.add_embeddings(embeddings=embeddings_per_image, image_name=image_name)

        self._embeddings_saver.save()
