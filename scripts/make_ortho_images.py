import logging

import click

import numpy as np

from imageio import imsave
from dtoolbioimage import ImageDataSet, zoom_to_match_scales


def make_cross_section_view(stack):

    xdim, ydim, zdim = stack.shape

    cr, cc, cz = map(lambda x: x//2, stack.shape)

    rslice = stack[cr, :, :]
    cslice = stack[:, cc, :]
    zslice = stack[:, :, cz]

    rslice_t = np.transpose(rslice)

    secondview = np.vstack([cslice, np.zeros((zdim, zdim), dtype=cslice.dtype)])
    firstview = np.vstack([zslice, rslice_t])

    orthoview = np.hstack([firstview, secondview])

    return orthoview


def make_cross_sections(image_ds):

    for im, sn in image_ds.iternames():

        logging.info('Generating view for {}'.format(im))
        stack = image_ds.get_stack(im, sn, channel=1)
        stack = zoom_to_match_scales(stack)

        orthoview = make_cross_section_view(stack)

        imsave('scratch/{}-orth.png'.format(im), orthoview)



@click.command()
@click.argument('image_dataset_uri')
def main(image_dataset_uri):

    logging.basicConfig(level=logging.INFO)

    image_ds = ImageDataSet(image_dataset_uri)
    make_cross_sections(image_ds)
    

if __name__ == "__main__":
    main()
