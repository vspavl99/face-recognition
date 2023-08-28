from abc import abstractmethod

from sklearn.cluster import KMeans, DBSCAN, MeanShift


class ClusterModel:
    @abstractmethod
    def predict(self, data):
        pass


class KMeansClustering(ClusterModel):
    def __init__(self, n_clusters=10):
        self._n_clusters = n_clusters
        self._model = KMeans(n_clusters=self._n_clusters, n_init='auto', random_state=2023)

    def predict(self, data):
        predictions = self._model.fit_predict(data)
        return predictions


class KMeansClusteringWithTuning(KMeansClustering):
    def __init__(self, tunner, n_clusters_limit=(10, 30)):
        super().__init__()

        k_means_tunner = tunner
        best_n_clusters = k_means_tunner.tune(n_clusters_limit=n_clusters_limit)
        self._model = KMeans(n_clusters=best_n_clusters, n_init='auto', random_state=2023)


class MeanShiftClustering(ClusterModel):
    def __init__(self, bandwidth=16):
        self._model = MeanShift(bandwidth=bandwidth)

    def predict(self, data):
        predictions = self._model.fit_predict(data)
        return predictions


class DBSCANClustering(ClusterModel):
    def __init__(self, *args, **kwargs):
        self._model = DBSCAN(*args, **kwargs)

    def predict(self, data):
        predictions = self._model.fit_predict(data)
        return predictions