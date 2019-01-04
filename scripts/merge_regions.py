import click

import numpy as np

from dtoolbioimage.segment import Segmentation3D


def merge_regions(segmentation_fpath, mergelist):

    segmentation = Segmentation3D.from_file(segmentation_fpath)

    for l1, l2 in mergelist:
        segmentation[np.where(segmentation == l2)] = l1

    segmentation.save('merged_05.tif')


@click.command()
@click.argument('segmentation_fpath')
def main(segmentation_fpath):

    mergelist = [(181, 91), (145, 138)]
    # mergelist = [(21, 66), (62, 41), (116, 195)]
    merge_regions(segmentation_fpath, mergelist)


if __name__ == "__main__":
    main()
