from typing import Protocol

import pandas as pd


class EmbeddingSaver(Protocol):
    def add_embeddings(self, embeddings, image_name: str):
        """Add embedding"""

    def save(self):
        """Save all embeddings"""


class EmbeddingSaverCSV(EmbeddingSaver):
    def __init__(self, csv_path='embeddings.csv'):
        """
        Save embedding CSV
        :param csv_path:
        :return:
        """
        self._embeddings = []
        self._csv_path = csv_path

    def add_embeddings(self, image_name: str, embeddings):
        self._embeddings.append((image_name, embeddings))

    def _save_file(self):
        dataframe = pd.DataFrame(data=self._embeddings, columns=['image_name', 'embeddings'])
        dataframe.to_csv(path_or_buf=self._csv_path, index=False)

    def save(self):
        self._save_file()
