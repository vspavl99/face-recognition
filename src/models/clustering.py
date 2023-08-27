from abc import abstractmethod
from sklearn.cluster import KMeans, DBSCAN


class ClusterModel:
    @abstractmethod
    def predict(self, data):
        pass


class KMeansClustering(ClusterModel):
    def __init__(self, n_clusters=10):
        self._n_clusters = n_clusters
        self._model = KMeans(n_clusters=self._n_clusters, n_init='auto')

    def predict(self, data):
        predictions = self._model.fit_predict(data)
        return predictions


class DBSCANClustering(ClusterModel):
    def __init__(self, *args, **kwargs):
        self._model = DBSCAN(*args, **kwargs)

    def predict(self, data):
        predictions = self._model.fit_predict(data)
        return predictions