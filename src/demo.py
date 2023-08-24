from src.cluster.find_similar import FindSimilar
from src.cluster.similarity import SimilarityEuclidean
from src.utils.embeddings_reader import EmbeddingReaderTXT
from src.utils.visualisation import vis_by_path
import umap

def resize_embedding(embeddings, dim):
    reducer = umap.UMAP(n_components=dim, random_state=2023)
    umap_emb = reducer.fit_transform(embeddings)
    return umap_emb

def find_similar_embeddings():
    emdeddings = EmbeddingReaderTXT(txt_path='data/processed/test-task/embeddings.txt').read()

    anchor_name = '0a184e3844b0470c96d737987497fb43.jpg'
    anchor = emdeddings[anchor_name]
    similarity = SimilarityEuclidean()
    find_similar = FindSimilar(anchor=anchor, similarity=similarity)
    vis_by_path(path=f'data/processed/test-task/clusters/{anchor_name}')

    for emb_name, emb in emdeddings.items():
        similarity = find_similar.check_neighborhoods(emb)
        if similarity < 25:
            print(emb_name, similarity)
            vis_by_path(path=f'data/processed/test-task/clusters/{emb_name}')

def clusters_str_to_id(dataframe):
    clusters_names = dataframe['cluster_id'].unique()
    _mapping = {}

    for i, cluster_name in enumerate(clusters_names):
        _mapping[cluster_name] = i

    dataframe['cluster_number'] = dataframe['cluster_id'].map(_mapping)
    return dataframe


def k_means():
    from sklearn.cluster import KMeans
    import numpy as np
    import pandas as pd
    from sklearn import metrics
    from src.eval.eval import EvalClusteringMetrics

    eval_metrics = EvalClusteringMetrics(
        metrics={
            "Rand_score": metrics.rand_score,
            "Homogeneity score": metrics.homogeneity_score,
            "Completeness score": metrics.completeness_score,
            "V-Measure score": metrics.v_measure_score
        }
    )

    data = pd.read_csv('data/processed/test-task/clusters.csv', index_col=0)
    data = clusters_str_to_id(data)
    embeddings = EmbeddingReaderTXT(txt_path='data/processed/test-task/embeddings.txt').read()

    image_names = list(embeddings.keys())
    embeddings = np.array(list(embeddings[image_name] for image_name in image_names))

    y_pred = KMeans(n_clusters=25, n_init='auto').fit_predict(embeddings)

    y_true = []
    for image_name in image_names:
        y_true.append(data[data['file_name'] == image_name]['cluster_number'].values[0])

    eval_metrics.compute_metrics(y_true, y_pred)

    resized_embeddings = resize_embedding(embeddings=embeddings, dim=2)
    import matplotlib.pyplot as plt

    plt.scatter(resized_embeddings[:, 0], resized_embeddings[:, 1], c=y_pred, cmap='Paired')
    plt.title('Predicted labels')
    plt.show()


    plt.scatter(resized_embeddings[:, 0], resized_embeddings[:, 1], c=y_true, cmap='Paired')
    plt.title('True labels')
    plt.legend()
    plt.show()

    import plotly.express as px

    fig = px.scatter(x=resized_embeddings[:, 0], y=resized_embeddings[:, 1])
    fig.show(renderer="svg")



if __name__ == '__main__':
    k_means()