from abc import abstractmethod

from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    @abstractmethod
    def transform(self, data):
        pass

class KMeansPreprocessor(DataPreprocessor):
    def __init__(self):
        self._scaler = StandardScaler()
    def transform(self, data):
        scaled_data = self._scaler.fit_transform(data)
        return scaled_data

class MeanShiftPreprocessor(DataPreprocessor):
    def transform(self, data):
        return data
