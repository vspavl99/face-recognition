from typing import Callable, Dict
from abc import abstractmethod


class EvalClusteringAbstract:
    @abstractmethod
    def compute_metrics(self, labels, predictions):
        pass


class EvalClustering(EvalClusteringAbstract):
    def __init__(self, metrics: Dict[str, Callable]):
        self._metrics = metrics

    def compute_metrics(self, labels, predictions):
        for metric_name, func in self._metrics.items():
            value = func(labels, predictions)
            print(f"{metric_name}: {value}")


class EvalClusteringTuner(EvalClusteringAbstract):
    def __init__(self, metric: Callable):
        self._metric = metric

    def compute_metrics(self, labels, predictions) -> float:
        return self._metric(labels, predictions)
