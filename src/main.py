import click

from src.cluster_service import ClusterService


@click.command()
@click.option('--path_to_images', type=str, help='Path to directory with images.')
@click.option('--path_to_target_clusters', type=str, help='Path csv file with target clusters.')
def main(path_to_images: str, path_to_target_clusters: str):
    service = ClusterService(path_to_images=path_to_images, path_to_target_clusters=path_to_target_clusters)
    service.run()


if __name__ == '__main__':
    main()