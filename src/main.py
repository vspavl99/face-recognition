from sklearn import metrics
import numpy as np

from src.features.preprocess import KMeansPreprocessor, MeanShiftPreprocessor
from src.eval.eval import EvalClustering
from src.utils.embeddings_reader import EmbeddingReaderTXT
from src.models.clustering import KMeansClustering, DBSCANClustering, MeanShiftClustering
from src.data.cluster_labels import ClusterLabelsCSV
from src.utils.image_iterator import ImagePathIterator
from src.models.emdedding_extractor import EmbeddingModelInsightface


class ClusterService:
    def __init__(self, path_to_images, path_to_target_clusters):
        """
        Class for clustering images in `path_to_images` directory. Metrics calculated according
        to `path_to_target_clusters` csv-file
        :param path_to_images: directory where images are located
        :param path_to_target_clusters: path to clusters.csv file
        """

        self._images_iter = ImagePathIterator(image_dir=path_to_images)
        self._model = EmbeddingModelInsightface()

        self._cluster_labels = ClusterLabelsCSV(file_path=path_to_target_clusters)
        self._data_preprocessor = MeanShiftPreprocessor()

        self._eval_metrics = EvalClustering(
            metrics={
                "Homogeneity score": metrics.homogeneity_score,
                "Completeness score": metrics.completeness_score,
                "V-Measure score": metrics.v_measure_score
            }
        )

        self._clustering_model = MeanShiftClustering()
        # self._clustering_model = KMeansClustering(n_clusters=25)


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

        embeddings = self.prepare_data(embeddings)

        return images_without_embeddings, images_with_embeddings, embeddings

    def _generate_embeddings_from_file(self):
        embedding_reader = EmbeddingReaderTXT(txt_path='data/processed/test-task/embeddings.txt')

        embeddings = embedding_reader.embedding_vectors
        embeddings = self.prepare_data(embeddings)

        images_with_embeddings = embedding_reader.embedding_names
        images_without_embeddings = set(self._cluster_labels.image_names) - set(images_with_embeddings)

        return images_without_embeddings, images_with_embeddings, embeddings

    def prepare_data(self, data):
        return self._data_preprocessor.transform(data)

    def run(self):

        # TODO: Generate embeddings by model self._generate_embeddings()
        images_without_embeddings, images_with_embeddings, embeddings = self._generate_embeddings_from_file()

        predictions_for_images_with_embeddings = self._clustering_model.predict(embeddings)

        predictions_for_images_without_embeddings = self._create_labels_for_background(
            max_label=np.max(predictions_for_images_with_embeddings), num_images=len(images_without_embeddings)
        )

        predictions = np.concatenate(
            (predictions_for_images_with_embeddings, predictions_for_images_without_embeddings)
        )

        labels = self._get_target(images_with_embeddings, images_without_embeddings)

        print('\nBackground and face images clustering: ')
        self._eval_metrics.compute_metrics(labels=labels, predictions=predictions)

        print('\nFace images clustering: ')
        self._eval_metrics.compute_metrics(
            labels=self._cluster_labels.get_labels_by_image_name(images_with_embeddings),
            predictions=predictions_for_images_with_embeddings
        )

    def _get_target(self, images_with_embeddings, images_without_embeddings):
        labels_for_images_with_embeddings = self._cluster_labels.get_labels_by_image_name(images_with_embeddings)
        labels_for_images_without_embeddings = self._cluster_labels.get_labels_by_image_name(images_without_embeddings)

        labels = np.concatenate(
            (labels_for_images_with_embeddings, labels_for_images_without_embeddings)
        )

        return labels

    @staticmethod
    def _create_labels_for_background(max_label, num_images):
        """
        Create constant labels for background
        :param max_label: max cluster label for non_background images
        :param num_images: length of vector for creating
        :return:
        """
        return np.ones(num_images) * (max_label + 1)


if __name__ == '__main__':
    service = ClusterService()
    service.run()