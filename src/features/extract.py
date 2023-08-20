import click

from src.features.emdedding_extractor import EmbeddingModelInsightface
from src.features.generate_embedding import FaceEmbeddingGenerator
from src.utils.embedding_saver import  EmbeddingSaverTXT
from src.utils.image_iterator import ImagePathIterator

@click.command()
@click.option('--data_dir', type=str, help='Path to a directory where images is located')
@click.option('--output_path', type=str, help='Path to a directory where output file will be saved')
def main(data_dir: str, output_path: str):
    images_iter = ImagePathIterator(image_dir=data_dir)
    model = EmbeddingModelInsightface()
    embedding_saver = EmbeddingSaverTXT(txt_path=output_path)

    face_embedding_generator = FaceEmbeddingGenerator(
        model=model, images_iter=images_iter, embeddings_saver=embedding_saver
    )
    face_embedding_generator.generate()

    print("Successfully saved")


if __name__ == '__main__':
    main()