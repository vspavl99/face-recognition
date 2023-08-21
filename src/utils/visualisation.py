import cv2
import matplotlib.pyplot as plt


def vis_by_path(path: str):

    image = cv2.imread(path)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()
