from pathlib import Path

import click

from skimage.restoration import denoise_tv_chambolle

from dtoolbioimage import ImageDataSet, Image3D, zoom_to_match_scales


def denoise_stack_from_dataset(imageds, image_name, series_name, channel, output_filename):

    stack = imageds.get_stack(image_name, series_name, channel)

    zoomed_stack = zoom_to_match_scales(stack)

    denoised_stack = denoise_tv_chambolle(zoomed_stack, weight=0.02)

    denoised_stack.view(Image3D).save(output_filename)


def derive_output_filename(output_dirpath, image_name, series_name):

    return output_dirpath/(image_name + '_denoised_venus.tif')


def denoise_all_stacks_in_dataset(imageds, channel, output_dirpath):

    for image_name in imageds.get_image_names():
        for series_name in imageds.get_series_names(image_name):
            output_filename = derive_output_filename(output_dirpath, image_name, series_name)
            denoise_stack_from_dataset(imageds, image_name, series_name, channel, output_filename)


@click.command()
@click.argument('dataset_uri')
def main(dataset_uri):

    imageds = ImageDataSet(dataset_uri)

    image_name = 'fca-3_FLC-Venus_root03'
    series_name = 'fca-3_FLC-Venus_root03 #1'
    channel = 0
    output_filename = 'root_03_denoised.tif'
    # denoise_stack_from_dataset(imageds, image_name, series_name, channel, output_filename)

    output_dirpath = Path('scratch/')
    denoise_all_stacks_in_dataset(imageds, channel, output_dirpath)


if __name__ == "__main__":
    main()
