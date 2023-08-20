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

    def read(self) -> dict:
        embeddings = {}

        with open(self._txt_path, 'r') as file:
            for line in file.readlines():
                image_name, embedding = line.split('\t')
                embeddings[image_name] = np.fromstring(string=embedding, dtype=float, sep=' ')

        return embeddings
