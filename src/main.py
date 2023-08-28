from sklearn import metrics
import numpy as np

from src.features.preprocess import KMeansPreprocessor
from src.eval.eval import EvalClustering
from src.utils.embeddings_reader import EmbeddingReaderTXT
from src.models.clustering import KMeansClustering, DBSCANClustering, MeanShiftClustering
from src.data.cluster_labels import ClusterLabelsCSV
from src.utils.image_iterator import ImagePathIterator
from src.models.emdedding_extractor import EmbeddingModelInsightface


class Service:
    def __init__(self):
        self._images_iter = ImagePathIterator(image_dir="/home/vpavlishen/data_ssd/vpavlishen/test-task/clusters")
        self._model = EmbeddingModelInsightface()

        self._cluster_labels = ClusterLabelsCSV(file_path='data/processed/test-task/clusters.csv')
        self._data_preprocessor = KMeansPreprocessor()

        self._eval_metrics = EvalClustering(
            metrics={
                "Homogeneity score": metrics.homogeneity_score,
                "Completeness score": metrics.completeness_score,
                "V-Measure score": metrics.v_measure_score
            }
        )

        self._clustering_model = KMeansClustering(n_clusters=25) # n_clusters is optimal value founded by tunner
        # self._clustering_model = DBSCANClustering()
        self._clustering_model = MeanShiftClustering()


    def _generate_embeddings(self):
        """
        Iterate over images through self._images_iter and produces embedding of faces
        :return: return three lists. name of images without face embeddings,
        with face embeddings, corresponding embeddings
        """
        images_without_embeddings, images_with_embeddings, embeddings = [], [], []
        for (image, image_name) in self._images_iter:
            embeddings_per_image = self._model.get_embeddings(image)

            if not embeddings_per_image:
                images_without_embeddings.append(image_name)
            else:
                images_with_embeddings.append(image_name)
                embeddings.append(embeddings_per_image)

        return images_without_embeddings, images_with_embeddings, embeddings

    def _generate_embeddings_from_file(self):
        embedding_reader = EmbeddingReaderTXT(txt_path='data/processed/test-task/embeddings.txt')
        images_with_embeddings = embedding_reader.embedding_names
        embeddings = embedding_reader.embedding_vectors
        images_without_embeddings = set(self._cluster_labels.image_names) - set(images_with_embeddings)
        return images_without_embeddings, images_with_embeddings, embeddings

    def prepare_data(self, data):
        return self._data_preprocessor.transform(data)

    def start(self):

        # TODO: Generate embeddings by model self._generate_embeddings()
        images_without_embeddings, images_with_embeddings, embeddings = self._generate_embeddings_from_file()

        labels_for_images_with_embeddings = self._cluster_labels.get_labels_by_image_name(images_with_embeddings)
        labels_for_images_without_embeddings = self._cluster_labels.get_labels_by_image_name(images_without_embeddings)

        # embeddings = self.prepare_data(embeddings)
        y_pred = self._clustering_model.predict(embeddings)
        print(np.unique(y_pred))
        #
        self._eval_metrics.compute_metrics(labels=labels_for_images_with_embeddings, predictions=y_pred)



if __name__ == '__main__':
    service = Service()
    service.start()