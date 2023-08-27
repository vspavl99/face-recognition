from sklearn import metrics

from src.eval.eval import EvalClustering
from src.utils.embeddings_reader import EmbeddingReaderTXT
from src.models.clustering import KMeansClustering
from src.data.cluster_labels import ClusterLabelsCSV


def main():
    eval_metrics = EvalClustering(
        metrics={
            "Rand_score": metrics.rand_score,
            "Homogeneity score": metrics.homogeneity_score,
            "Completeness score": metrics.completeness_score,
            "V-Measure score": metrics.v_measure_score
        }
    )

    cluster_labels = ClusterLabelsCSV(file_path='data/processed/test-task/clusters.csv')

    clustering_model = KMeansClustering(n_clusters=25)
    embedding_reader = EmbeddingReaderTXT(txt_path='data/processed/test-task/embeddings.txt')

    image_names = embedding_reader.embedding_names
    embeddings = embedding_reader.embedding_vectors

    y_pred = clustering_model.predict(embeddings)
    y_true = cluster_labels.get_labels_by_image_name(image_names)

    eval_metrics.compute_metrics(labels=y_true, predictions=y_pred)


if __name__ == '__main__':
    main()