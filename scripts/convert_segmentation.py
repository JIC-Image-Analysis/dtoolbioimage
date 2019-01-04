"""Convert segmentation from RGB encoding to 32 bit TIFF"""

import click

import numpy as np

from dtoolbioimage.segment import Segmentation3D


def convert_segmentation(input_fpath, output_fpath, output_encoding):

    segmentation = Segmentation3D.from_file(input_fpath)

    segmentation.save(output_fpath, encoding=output_encoding)


@click.command()
@click.argument('input_fpath')
@click.argument('output_fpath')
def main(input_fpath, output_fpath):

    convert_segmentation(input_fpath, output_fpath, '32bit')


if __name__ == "__main__":
    main()
