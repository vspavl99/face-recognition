from abc import abstractmethod

class EvalClustering:
    @abstractmethod
    def compute_metrics(self, y_true, y_pred):
        pass