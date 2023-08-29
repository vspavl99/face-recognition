import cv2
import matplotlib.pyplot as plt


def vis_by_path(path: str):

    image = cv2.imread(path)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()


def vis_clusters(projected_embeddings, predicted_clusters, target_clusters):

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(projected_embeddings[:, 0], projected_embeddings[:, 1], c=predicted_clusters, cmap='Paired')
    plt.title('Predicted clusters')

    plt.subplot(1, 2, 2)
    plt.scatter(projected_embeddings[:, 0], projected_embeddings[:, 1], c=target_clusters, cmap='Paired')
    plt.title('Target clusters')
    plt.show()
