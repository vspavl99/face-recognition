from sklearn import metrics

from src.eval.eval import EvalClustering
from src.utils.embeddings_reader import EmbeddingReaderTXT
from src.models.clustering import KMeansClustering
from src.data.cluster_labels import ClusterLabelsCSV


def find_best_n_clusters():
    eval_metrics = EvalClustering(
        metrics={
            "Homogeneity score": metrics.homogeneity_score,
            "Completeness score": metrics.completeness_score,
            "V-Measure score": metrics.v_measure_score
        }
    )
    cluster_labels = ClusterLabelsCSV(file_path='data/processed/test-task/clusters.csv')
    embedding_reader = EmbeddingReaderTXT(txt_path='data/processed/test-task/embeddings.txt')
    image_names = embedding_reader.embedding_names
    embeddings = embedding_reader.embedding_vectors
    y_true = cluster_labels.get_labels_by_image_name(image_names)

    K = range(1, 50)
    best_score, best_value = 0, 0

    for k in K:
        clustering_model = KMeansClustering(n_clusters=k)
        y_pred = clustering_model.predict(embeddings)
        scores = eval_metrics.compute_metrics(labels=y_true, predictions=y_pred)

        if scores['V-Measure score'] > best_score:
            best_score = scores['V-Measure score']
            best_value = k

    print(best_score, best_value)


if __name__ == '__main__':
    find_best_n_clusters()