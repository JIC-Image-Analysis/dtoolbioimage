import os

from io import BytesIO
from collections import defaultdict

from imageio import imread, imsave, mimsave

import numpy as np

import dtoolcore


def scale_to_uint8(array):

    scaled = array.astype(np.float32)
    scaled = 255 * (scaled - scaled.min()) / (scaled.max() - scaled.min())

    return scaled.astype(np.uint8)


class ImageMetadata(object):

    def __init__(self, metadata_dict):
        self.metadata_dict = metadata_dict

    def __getattr__(self, name):
        return self.metadata_dict[name]


class Image3D(np.ndarray):

    def _repr_png_(self):

        if len(self.shape) == 2:
            b = BytesIO()
            scaled = scale_to_uint8(self)
            imsave(b, scaled, 'PNG')

            return b.getvalue()

        return self[:, :,0]._repr_png_()

    def save(self, fpath):
        _, ext = os.path.splitext(fpath)
        assert ext in ['.tif', '.tiff']

        # We use row, col, z, but mimsave expects z, row, col
        transposed = np.transpose(self, axes=[2, 0, 1])
        mimsave(fpath, transposed)



class ImageDataSet(object):

    def __init__(self, uri):
        self.dataset = dtoolcore.DataSet.from_uri(uri)

        self.build_index()

        self.metadata = self.dataset.get_overlay('microscope_metadata')

    def build_index(self):
        coords_overlay = self.dataset.get_overlay("plane_coords")

        def specifier_tuple(idn):
            relpath = self.dataset.item_properties(idn)["relpath"]
            image_name, series_name, _ = relpath.split('/')
            channel = int(coords_overlay[idn]['C'])
            plane = int(coords_overlay[idn]['Z'])

            return image_name, series_name, channel, plane, idn

        specifiers = map(specifier_tuple, self.dataset.identifiers)

        # Nested dictionary of dictionaries (x4)
        planes_index = defaultdict(lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))))
        for image_name, series_name, channel, plane, idn in specifiers:
            planes_index[image_name][series_name][channel][plane] = idn

        self.planes_index = planes_index

    def get_stack(self, image_name, series_name, channel=0):

        z_idns = self.planes_index[image_name][series_name][channel]

        images = [
            imread(self.dataset.item_content_abspath(z_idns[z]))
            for z in sorted(z_idns)
        ]

        def safe_select_channel(im):
            if len(im.shape) == 2:
                return im
            elif len(im.shape) == 3:
                return im[:, :, channel]
            else:
                raise("Weird dimensions")

        selected_planes = [safe_select_channel(im) for im in images]

        stack = np.dstack(selected_planes).view(Image3D)
        stack.metadata = ImageMetadata(self.metadata[z_idns[0]])

        return stack

    def get_image_names(self):
        
        return list(self.planes_index.keys())

    def get_series_names(self, image_name):

        return list(self.planes_index[image_name].keys())
