import logging

import click

from dtoolbioimage import ImageDataSet

@click.command()
@click.argument('ids_uri')
@click.argument('image_name')
@click.argument('series_name')
@click.argument('output_fpath')
@click.option('--series-index', type=int, default=0)
def main(ids_uri, image_name, series_name, output_fpath, series_index):

    logging.basicConfig(level=logging.INFO)

    ids = ImageDataSet(ids_uri)

    stack = ids.get_stack(image_name, series_name, series_index)

    logging.info(f"Loaded stack with shape {stack.shape}")

    stack.save(output_fpath)


if __name__ == "__main__":
    main()
