from typing import Protocol

import pandas as pd


class EmbeddingSaver(Protocol):
    def add_embeddings(self, embeddings, image_name: str):
        """Add embedding"""

    def save(self):
        """Save all embeddings"""


class EmbeddingSaverCSV(EmbeddingSaver):
    def __iter__(self, csv_path='embeddings.csv'):
        """
        Save embedding CSV
        :param csv_path:
        :return:
        """
        self._embeddings = []
        self._csv_path = csv_path

    def add_embeddings(self, embeddings, image_name: str):
        self._embeddings.append((image_name, embeddings))

    def _save_file(self):
        dataframe = pd.DataFrame(data=self._embeddings, columns=['embeddings', 'image_name'])
        dataframe.to_csv(self._csv_path)

    def save(self):
        self._save_file()
