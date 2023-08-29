from abc import abstractmethod

import pandas as pd
import numpy as np


class ClusterLabels:
    @abstractmethod
    def cluster_id(self):
        pass


class ClusterLabelsCSV(ClusterLabels):
    """
    Reads true cluster labels from a csv file. Prepare dataframe for further processing
    """
    def __init__(self, file_path: str):
        self._csv_path = file_path
        self._dataframe = self._read_csv()

    def _read_csv(self) -> pd.DataFrame:
        data = pd.read_csv(self._csv_path, index_col=0)
        data = self._convert_cluster_id(data)
        return data

    @staticmethod
    def _convert_cluster_id(dataframe) -> pd.DataFrame:
        """
        Add to dataframe new column "cluster_number", where cluster_id is just the number
        :param dataframe: dataframe with cluster_id column
        :return: dataframe with new columns "cluster_number"
        """
        assert 'cluster_id' in dataframe.columns

        cluster_ids = dataframe['cluster_id'].unique()
        id_to_number = {}

        for index, cluster_id in enumerate(cluster_ids):
            id_to_number[cluster_id] = index

        dataframe['cluster_number'] = dataframe['cluster_id'].map(id_to_number)
        return dataframe


    def get_labels_by_image_name(self, image_names) -> list:
        """
        Return cluster_number for corresponding images
        :param image_names: name of images for filtering labels
        :return: labels of clusters in corresponding order by image_names
        """
        cluster_numbers = []
        for image_name in image_names:
            cluster_numbers.append(
                self._dataframe[self._dataframe['file_name'] == image_name]['cluster_number'].values[0]
            )

        return cluster_numbers

    @property
    def cluster_id(self) -> np.ndarray:
        return self._dataframe['cluster_id'].values

    @property
    def cluster_number(self) -> np.ndarray:
        return self._dataframe['cluster_number'].values

    @property
    def image_names(self) -> np.ndarray:
        return self._dataframe['file_name'].values