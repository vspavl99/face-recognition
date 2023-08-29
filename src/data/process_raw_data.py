from pathlib import Path
import zipfile

import click


@click.command()
@click.option('--raw_data_path', type=str, help='Path to a archived data file.')
@click.option('--save_data_dir', type=str, help='Path to a directory where to save prepared data')
def main(raw_data_path: str, save_data_dir: str):

    raw_data_path = Path(raw_data_path)
    save_data_dir = Path(save_data_dir)

    unzip_archive(raw_data_path=raw_data_path, save_data_dir=save_data_dir)


def unzip_archive(raw_data_path: Path, save_data_dir: Path):
    with zipfile.ZipFile(file=raw_data_path, mode='r') as zip_ref:
        save_data_dir.mkdir(exist_ok=True)
        zip_ref.extractall(save_data_dir)


if __name__ == '__main__':
    main()
