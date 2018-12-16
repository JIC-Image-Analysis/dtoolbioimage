import os

import numpy as np
import SimpleITK as sitk

from imageio import mimsave, volread

from dtoolbioimage.ipyutils import cached_segmentation_viewer
from dtoolbioimage.util.array import pretty_color_array, unique_color_array

from skimage.measure import regionprops

from scipy.ndimage.morphology import binary_erosion


def select_region(segmentation, label):
    by_label = {r.label: r for r in regionprops(segmentation)}

    rmin, cmin, zmin, rmax, cmax, zmax = by_label[label].bbox

    selected = segmentation[rmin:rmax, cmin:cmax, zmin:zmax]   

    return selected == label


def spherality(region):
    eroded = binary_erosion(region)
    surface = region ^ eroded

    S = np.sum(surface)
    V = np.sum(region)

    mult = 4.5 * np.sqrt(np.pi)

    return mult * V / np.power(S, 1.5)


class Segmentation3D(np.ndarray):

    def save(self, fpath):

        _, ext = os.path.splitext(fpath)
        assert ext in ['.tif', '.tiff']

        uci = self.unique_color_image

        transposed = np.transpose(uci, [2, 0, 1, 3])

        mimsave(fpath, transposed)

    @property
    def unique_color_image(self):

        return unique_color_array(self)
    
    @property
    def pretty_color_image(self):

        return pretty_color_array(self)

    def _ipython_display_(self):

        display(cached_segmentation_viewer(self))

    @classmethod
    def from_file(cls, fpath):

        unique_color_image = volread(fpath)

        zdim, xdim, ydim, _ = unique_color_image.shape

        planes = []
        for z in range(zdim):
            segmentation = np.zeros((xdim, ydim), dtype=np.uint32)
            segmentation += unique_color_image[z,:,:,2]
            segmentation += unique_color_image[z,:,:,1] * 256
            segmentation += unique_color_image[z,:,:,0] * 256 * 256
            planes.append(segmentation)

        return np.dstack(planes).view(Segmentation3D)

    @property
    def labels(self):
        return set(np.unique(self)) - set([0])

    def region_in_bb(self, label):
        by_label = {r.label: r for r in regionprops(self)}

        rmin, cmin, zmin, rmax, cmax, zmax = by_label[label].bbox

        selected = self[rmin:rmax, cmin:cmax, zmin:zmax]

        return selected == label


def sitk_watershed_segmentation(stack):
    """Segment the given stack."""

    itk_im = sitk.GetImageFromArray(stack)
    median_filtered = sitk.Median(itk_im)
    grad_mag = sitk.GradientMagnitude(median_filtered)
    blurred = sitk.DiscreteGaussian(grad_mag, 2.0)
    segmentation = sitk.MorphologicalWatershed(blurred, level=0.664)
    relabelled = sitk.RelabelComponent(segmentation)

    return sitk.GetArrayFromImage(relabelled).view(Segmentation3D)


def filter_segmentation_by_size(segmentation):

    filtered = segmentation.copy()

    filtered[np.where(filtered > 150)] = 0
    filtered[np.where(filtered < 3)] = 0

    return filtered.view(Segmentation3D)
