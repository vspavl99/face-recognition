from sklearn.metrics import v_measure_score

from src.utils.embeddings_reader import EmbeddingReaderTXT
from src.data.cluster_labels import ClusterLabelsCSV
from src.eval.eval_clustering import EvalClusteringTuner
from src.models.tuners import MeanShiftTunner, KMeansTunner


def embeddings_and_labels_init():
    _cluster_labels = ClusterLabelsCSV(file_path='data/processed/test-task/clusters.csv')

    embedding_reader = EmbeddingReaderTXT(txt_path='data/processed/test-task/embeddings.txt')
    images_with_embeddings = embedding_reader.embedding_names
    embeddings = embedding_reader.embedding_vectors

    labels_for_images_with_embeddings = _cluster_labels.get_labels_by_image_name(images_with_embeddings)
    return embeddings, labels_for_images_with_embeddings


def tune_k_means():
    embeddings, labels_for_images_with_embeddings = embeddings_and_labels_init()

    tuner = KMeansTunner(
        evaluator=EvalClusteringTuner(v_measure_score),
        data=embeddings,
        target=labels_for_images_with_embeddings
    )

    print(tuner.tune())


def tune_mean_shift():
    embeddings, labels_for_images_with_embeddings = embeddings_and_labels_init()

    tuner = MeanShiftTunner(
        evaluator=EvalClusteringTuner(v_measure_score),
        data=embeddings,
        target=labels_for_images_with_embeddings
    )

    print(tuner.tune())


if __name__ == '__main__':
    print("K means n_cluster ", tune_k_means())
    print("MeanShift bandwidth ", tune_mean_shift())