import click

from dtoolbioimage import ImageDataSet


def show_summary(image_dataset):

    def get_dimension_data(entry):
        im_name, s_name = entry
        channels = image_dataset.planes_index[im_name][s_name].keys()
        n_channels = len(channels)
        n_planes = len(image_dataset.planes_index[im_name][s_name][0])
        idn = image_dataset.planes_index[im_name][s_name][0][0]
        xdim = image_dataset.metadata[idn]['SizeX']
        ydim = image_dataset.metadata[idn]['SizeY']

        return (n_channels, n_planes, xdim, ydim)

    for entry in image_dataset.iternames():
        im_name, s_name = entry
        n_channels, n_planes, xdim, ydim = get_dimension_data(entry)
        print("{}, {}, {} channels, {}x{}x{}".format(im_name, s_name, n_channels, xdim, ydim, n_planes))



@click.command()
@click.argument('image_dataset_uri')
def main(image_dataset_uri):

    ids = ImageDataSet(image_dataset_uri)
    show_summary(ids)


if __name__ == "__main__":
    main()
