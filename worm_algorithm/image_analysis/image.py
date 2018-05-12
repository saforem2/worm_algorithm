import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import math
import itertools

def show(image):
    fig, ax = plt.subplots()
    img = ax.imshow(image, cmap='Greys', aspect=1, interpolation='none')
    plt.colorbar(img)
    return fig, ax

class Image(object):
    """Class to identify boundaries between black and white pixels in image.
    """
    def __init__(self, image_file):
        #self.cutoff = cutoff
        try:
            self._image_orig = scipy.misc.imread(image_file, "mode=L")
        except FileNotFoundError:
            raise f"Unable to load from {image_file}"
        self._Nx, self._Ny = self._image_orig.shape
        self._image_flat = self._image_orig.flatten()
        self._image = np.reshape(1 - self._image_flat / 255.0,
                                 (self._Nx, self._Ny))
        self._image_cropped = None
        self._image_bw = None
        self._image_links = None
        #self._original = self._image.copy()

    #  def show(self, image=None):
    #      """Create figure containing image, return figure and axes instances."""
    #      if image is None:
    #          image = self._image
    #      fig, ax = plt.subplots()
    #      img = ax.imshow(image, cmap='Greys', aspect=1,
    #                      interpolation='none')
    #      plt.colorbar(img)
    #      return fig, ax

    def crop(self, x_start, y_start, x_end, y_end):
        """Crop image from (x_start, y_start) to (x_end, y_end)."""
        cropped_image = self._image[x_start:x_end, y_start:y_end]
        self._image_cropped = cropped_image
        #self._image = cropped_image
        return cropped_image

    def _cutoff(self, cutoff, image=None):
        """Convert image from greyscale to exclusively black and white,
        determined by the value of cutoff."""
        if image is None:
            image = self._image
        image_flat = np.ndarray.flatten(image)
        Nx, Ny = image.shape
        #n_sites = image.shape[0] * image.shape[1]
        image_bw = np.array([-1 if i < cutoff else 1 for i in image_flat])
        image_bw = (np.reshape(image_bw, (Nx, Ny)) + 1) / 2.0
        self._image_bw = image_bw
        return image_bw

    def get_boundaries(self, cutoff, image=None):
        """Identify boundaries separating black and white regions of image."""
        if image is None:
            image = self._image
        image_bw = self._cutoff(cutoff, image)
        Nx, Ny = image_bw.shape
        links_arr = np.zeros((Nx, Ny, 2))
        for x, y in itertools.product(range(Nx), range(Ny)):
            links_arr[(x+1)%Nx, y, 1] = (int(round(image_bw[x, y]))
                                         + int(round(image_bw[(x+1)%Nx, y])))%2
            links_arr[x, (y+1)%Ny, 0] = (int(round(image_bw[x, y]))
                                         + int(round(image_bw[x, (y+1)%Ny])))%2
        image_links = np.zeros((2*Nx, 2*Ny), dtype=int)
        for x, y in itertools.product(range(Nx), range(Ny)):
            image_links[2*x+1, 2*y] = links_arr[x, y, 0]
            image_links[2*x, 2*y+1] = links_arr[x, y, 1]
            link_sum = (links_arr[x, y, 0] + links_arr[x, y, 1]
                        + links_arr[(x-1)%Nx, y, 0] + links_arr[x, (y-1)%Ny, 1])
            if link_sum != 0:
                image_links[2*x, 2*y] = 1
        self._image_links = image_links
        return image_links
