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
        Save embedding in CSV file
        :param csv_path: path where embedding csv file will be saved
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


class EmbeddingSaverTXT(EmbeddingSaver):
    def __init__(self, txt_path='embeddings.txt'):
        """
        Save embeddings in txt file
        :param txt_path: path where embedding txt file will be saved
        :return:
        """
        self._embeddings = []
        self._txt_path = txt_path

    def add_embeddings(self, image_name: str, embeddings):
        for embedding in embeddings:
            self._embeddings.append(f"{image_name} \t {self._emb_vector_to_str(embedding)} \n")

    @staticmethod
    def _emb_vector_to_str(embedding):
        return " ".join([str(value) for value in embedding])

    def _save_file(self):
        with open(self._txt_path, 'w') as file:
            file.writelines(self._embeddings)

    def save(self):
        self._save_file()