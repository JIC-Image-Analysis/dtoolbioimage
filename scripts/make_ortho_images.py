import logging

import click

import numpy as np

from imageio import imsave
from dtoolbioimage import ImageDataSet, zoom_to_match_scales
from dtool_utils.derived_dataset import DerivedDataSet


def make_cross_section_view(stack):

    zdim = stack.shape[2]

    cr, cc, cz = map(lambda x: x//2, stack.shape)

    rslice = stack[cr, :, :]
    cslice = stack[:, cc, :]
    zslice = stack[:, :, cz]

#     zslice[cr,:] = 255
#     zslice[:,cc] = 255
#     cslice[:,cz] = 255

    rslice_t = np.transpose(rslice)

    secondview = np.vstack([cslice, np.zeros((zdim, zdim), dtype=cslice.dtype)])
    firstview = np.vstack([zslice, rslice_t])

    orthoview = np.hstack([firstview, secondview])

    return orthoview


def make_cross_sections(image_ds, output_ds):

    for im_name, sn in image_ds.iternames():

        logging.info('Generating view for {}'.format(im_name))
        stack = image_ds.get_stack(im_name, sn, channel=1)
        stack = zoom_to_match_scales(stack)

        orthoview = make_cross_section_view(stack)

        relpath = f"{im_name}-orth.png"
        fpath = output_ds.staging_fpath(relpath)
        imsave(fpath, orthoview)



@click.command()
@click.argument('image_dataset_uri')
@click.argument('output_base_uri')
@click.argument('output_name')
def main(image_dataset_uri, output_base_uri, output_name):

    logging.basicConfig(level=logging.INFO)

    image_ds = ImageDataSet(image_dataset_uri)

    with DerivedDataSet(output_base_uri, output_name, image_ds) as output_ds:
        make_cross_sections(image_ds, output_ds)
    

if __name__ == "__main__":
    main()
