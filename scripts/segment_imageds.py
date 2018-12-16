import click

from dtoolbioimage import ImageDataSet, zoom_to_match_scales
from dtoolbioimage.segment import sitk_watershed_segmentation, filter_segmentation_by_size


def segment_image_from_dataset(imageds, image_name, series_name, wall_channel, output_filename):

    wall_stack = imageds.get_stack(image_name, series_name, wall_channel)

    zoomed_wall_stack = zoom_to_match_scales(wall_stack)

    segmentation = sitk_watershed_segmentation(zoomed_wall_stack)

    segmentation.save(output_filename)


@click.command()
@click.argument('dataset_uri')
def main(dataset_uri):

    imageds = ImageDataSet(dataset_uri)

    image_name = 'fca-3_FLC-Venus_root02'
    series_name = 'fca-3_FLC-Venus_root02 #1'
    wall_channel = 1
    output_filename = 'root_02_segmentation.tif'
    segment_image_from_dataset(imageds, image_name, series_name, wall_channel, output_filename)

if __name__ == "__main__":
    main()
