from pathlib import Path
import numpy as np
import cv2


class ImagePathIterator:
    def __init__(self, image_dir: str):
        """
        Class for iterating over images in a given directory.
        :param image_dir: path to directory with images
        """
        self._image_dir = Path(image_dir)
        self._iter_image_paths =  self._image_dir.iterdir()

    def __iter__(self):
        return self

    @staticmethod
    def _get_image(path) -> np.ndarray:
        image = cv2.imread(str(path))
        return image

    def __next__(self):
        image_path = next(self._iter_image_paths)
        image = self._get_image(image_path)
        return image, image_path.name
