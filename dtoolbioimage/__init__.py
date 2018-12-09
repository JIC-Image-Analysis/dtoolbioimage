import os

from io import BytesIO
from collections import defaultdict

from imageio import imread, imsave, mimsave

import numpy as np

import dtoolcore

from scipy.ndimage import zoom

from dtoolbioimage.ipyutils import simple_stack_viewer

from IPython.display import display


def zoom_to_match_scales(stack):
    px = float(stack.metadata.PhysicalSizeX)
    pz = float(stack.metadata.PhysicalSizeZ)
    ratio = pz / px
    zoomed = zoom(stack, (1, 1, ratio))

    zoomed_image = zoomed.view(Image3D)
    zoomed_image.metadata = stack.metadata

    return zoomed_image


def autopad(stack):
    xdim, ydim, zdim = stack.shape
    assert(xdim == ydim)

    n_pad_before = (xdim - zdim) // 2
    n_pad_after = xdim - (n_pad_before + zdim)

    zeros = np.zeros((xdim, ydim), dtype=np.uint8)
    pad_before = np.dstack([zeros] * n_pad_before)
    pad_after = np.dstack([zeros] * n_pad_after)

    return np.dstack((pad_before, stack, pad_after))


def scale_to_uint8(array):

    scaled = array.astype(np.float32)

    if scaled.max() - scaled.min() == 0:
        return np.zeros(array.shape, dtype=np.uint8)

    scaled = 255 * (scaled - scaled.min()) / (scaled.max() - scaled.min())

    return scaled.astype(np.uint8)


class ImageMetadata(object):

    def __init__(self, metadata_dict):
        self.metadata_dict = metadata_dict

    def __getattr__(self, name):
        return self.metadata_dict[name]


class Image(np.ndarray):

     def _repr_png_(self):

        b = BytesIO()
        scaled = scale_to_uint8(self)
        imsave(b, scaled, 'PNG')

        return b.getvalue()


class Image3D(np.ndarray):

    def _ipython_display_(self):

        if len(self.shape) == 2:
            display(self.view(Image))
            return

        display(simple_stack_viewer(self))

    def _repr_png_(self):

        if len(self.shape) == 2:
            b = BytesIO()
            scaled = scale_to_uint8(self)
            imsave(b, scaled, 'PNG')

            return b.getvalue()

        return simple_stack_viewer(self)

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
                maxvals = list(im[:,:,c].max() for c in range(3))
                c = maxvals.index(max(maxvals))
                return im[:, :, c]
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