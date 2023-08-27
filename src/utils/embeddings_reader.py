from typing import Protocol

import numpy as np


class EmbeddingReader(Protocol):
    def read(self) -> dict:
        """Read all embeddings"""

class EmbeddingReaderTXT(EmbeddingReader):
    def __init__(self, txt_path='embeddings.txt'):
        """
        Read embeddings in txt file
        :param txt_path: path where embedding txt file located
        :return:
        """
        self._txt_path = txt_path

        self._embeddings = {}
        self.read()

    def read(self) -> dict:

        with open(self._txt_path, 'r') as file:
            for line in file.readlines():
                image_name, embedding = line.split('\t')
                self._embeddings[image_name] = np.fromstring(string=embedding, dtype=float, sep=' ')

        return self._embeddings

    @property
    def embedding_names(self) -> list:
        return list(self._embeddings.keys())

    @property
    def embedding_vectors(self) -> np.array:
        return np.array(list(self._embeddings.values()))
