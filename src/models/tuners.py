from abc import abstractmethod

from sklearn.cluster import MeanShift

from src.eval.eval_clustering import EvalClusteringAbstract
from src.models.clustering import KMeansClustering


class ParamsTunner:
    @abstractmethod
    def tune(self):
        pass


class KMeansTunner(ParamsTunner):
    def __init__(self, evaluator: EvalClusteringAbstract, data, target):
        self._evaluator = evaluator

        self._data = data
        self._target = target

    def tune(self, n_clusters_limit=(10, 30)):

        best_score, best_value = 0, 0

        for k in range(*n_clusters_limit):
            clustering_model = KMeansClustering(n_clusters=k)
            y_pred = clustering_model.predict(self._data)
            current_score = self._evaluator.compute_metrics(labels=self._target, predictions=y_pred)

            if current_score > best_score:
                best_score = current_score
                best_value = k

        return best_value


class MeanShiftTunner(ParamsTunner):
    def __init__(self, evaluator: EvalClusteringAbstract, data, target):
        self._evaluator = evaluator

        self._data = data
        self._target = target

    def tune(self, bandwidth_limit=(10, 30)):

        best_score, best_value = 0, 0

        for k in range(*bandwidth_limit):
            y_pred = MeanShift(bandwidth=k).fit_predict(self._data)
            current_score = self._evaluator.compute_metrics(labels=self._target, predictions=y_pred)

            if current_score > best_score:
                best_score = current_score
                best_value = k

        return best_value
