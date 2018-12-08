
from collections import defaultdict

from imageio import imread

import numpy as np

import dtoolcore

class ImageDataSet(object):

    def __init__(self, uri):
        self.dataset = dtoolcore.DataSet.from_uri(uri)

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

        selected_planes = [im[:, :, channel] for im in images]

        return np.dstack(selected_planes)
