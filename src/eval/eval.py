from typing import Callable, List, Dict
from abc import abstractmethod


class EvalClusteringAbstract:
    @abstractmethod
    def compute_metrics(self, y_true, y_pred):
        pass


class EvalClustering(EvalClusteringAbstract):
    def __init__(self, metrics: Dict[str, Callable]):
        self._metrics = metrics

    def compute_metrics(self, labels, predictions):
        for metric_name, func in self._metrics.items():
            value = func(labels, predictions)
            print(f"{metric_name}: {value}")