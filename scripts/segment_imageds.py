from pathlib import Path


import click

from dtoolbioimage import ImageDataSet, zoom_to_match_scales
from dtoolbioimage.segment import sitk_watershed_segmentation, filter_segmentation_by_size


def derive_output_filename(output_dirpath, image_name, series_name):

    return output_dirpath/(image_name + '_segmentation.tif')


def segment_image_from_dataset(imageds, image_name, series_name, wall_channel, output_filename):

    wall_stack = imageds.get_stack(image_name, series_name, wall_channel)

    zoomed_wall_stack = zoom_to_match_scales(wall_stack)

    segmentation = sitk_watershed_segmentation(zoomed_wall_stack)

    filtered_segmentation = filter_segmentation_by_size(segmentation, 200)

    filtered_segmentation.save(output_filename)


def segment_all_images_in_dataset(imageds, channel, output_dirpath):
    for image_name in imageds.get_image_names():
        for series_name in imageds.get_series_names(image_name):
            output_filename = derive_output_filename(output_dirpath, image_name, series_name)
            segment_image_from_dataset(imageds, image_name, series_name, channel, output_filename)


@click.command()
@click.argument('dataset_uri')
def main(dataset_uri):

    imageds = ImageDataSet(dataset_uri)

    # image_name = 'fca-3_FLC-Venus_root02'
    # series_name = 'fca-3_FLC-Venus_root02 #1'
    wall_channel = 1
    # output_filename = 'root_02_segmentation.tif'
    # segment_image_from_dataset(imageds, image_name, series_name, wall_channel, output_filename)
    segment_all_images_in_dataset(imageds, wall_channel, Path('scratch/'))

if __name__ == "__main__":
    main()
