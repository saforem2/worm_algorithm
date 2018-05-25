import numpy as np 
from utils.utils import *

class CountBonds(object):
    """Class to obtain statistics about the average number of bonds <Nb> and
    the variance in the average number of bonds, <Nb^2> - <Nb>^2

    Args:
        image_set (array-like):
            Array of images for which <Nb> and <Delta_{N_b}^2> are calculated
            and averaged over.
        num_blocks (int, default=20):
            Number of blocks to be used for block resampling for bootstrap
            error analysis.
        save (bool, default=False):
            Whether or not to save the resulting bond_statistics data.
        verbose (bool, default=False):
            Whether or not to display information as the analysis is being
            performed.
    """
    def __init__(self, image_set, num_blocks=20, save=False,
                 verbose=False):
        #  if len(image_set.shape) == 1:
        #      self.reshape_image_set()
        self._image_set = image_set
        self._num_images, self._Lx, self._Ly = self._image_set.shape
        #self._block_val = block_val
        self._num_blocks = num_blocks
        self._verbose = verbose
        #  self.bond_stats = None
        #  if block_val is None:
        #      self._wx = 2 * self._Lx
        #      self._wy = 2 * self._Ly

    @staticmethod
    def reshape_image_set(image_set):
        """Method for reshaping images into a 2D array of pixels, if initally
        provided as flattened array."""
        pass

    def _count_bonds(self, image):
        """ Count bonds on single image, and return Nb. """
        if image.shape != (self._Lx, self._Ly):
            try:
                image = image.reshape(self._Lx, self._Ly)
            except ValueError:
                raise "Unable to properly reshape image."
        #Nb = np.sum(image)
        bond_idxs = [(i, j) for i in range(self._Lx) for j in range(self._Ly)
                     if (i + j) % 2 == 1]
        Nb = np.sum([image[i] for i in bond_idxs])
        return Nb

    def _calc_averages(self, data_set=None):
        """Calculate <Nb> and <Delta_{Nb}^2> by running _count_bonds method on
        each image in self._image_set."""
        if data_set is None:
            data_set = self._image_set
        Nb_arr = np.array([self._count_bonds(image)
                           for image in data_set])
        Nb2_arr = Nb_arr ** 2
        Nb_avg = np.mean(Nb_arr)
        Nb2_avg = np.mean(Nb2_arr)
        Nb_avg2 = Nb_avg ** 2
        delta_Nb2 = Nb2_avg - Nb_avg2
        return Nb_avg, delta_Nb2

    def _count_bonds_with_err(self):
        """Calculate the average number of active bonds (Nb) for the boundary
        images using the previously defined methods."""
        bond_stats = self._calc_averages()
        data_rs = block_resampling(self._image_set, self._num_blocks)
        bond_stats_rs = []
        err = []

        for block in data_rs:
            bond_stats_rs.append(self._calc_averages(block))
        bond_stats_rs = np.array(bond_stats_rs)
        for idx in range(len(bond_stats)):
            _err = jackknife_err(y_i=bond_stats_rs[:, idx],
                                 y_full = bond_stats[idx],
                                 num_blocks=self._num_blocks)
            err.append(_err)
        return bond_stats, err

    def count_bonds(self):
        """Calculate bond statistics for entirety of self._image set data,
        including error analysis."""
        val, err = self._count_bonds_with_err()
        bond_stats = np.array([val[0], err[0], val[1], err[1]])
        #bond_stats.append(np.array([val[0], err[0], val[1], err[1]]))
        return bond_stats









#  def count_bonds(image_set):
#      """Method for counting 'bonds' in a set of worm-type images."""
#      w = image_set[0].shape[0]
#      bond_idxs = [(i, j) for i in range(w)
#                   for j in range(w)
#                   if (i + j) % 2 == 1]
#      bc_arr = np.array([np.sum([image[i] for i in bond_idxs])
#                         for image in image_set])
#      bc2_arr = bc_arr ** 2
#      Nb_avg = np.mean(bc_arr)
#      Nb2_avg = np.mean(bc2_arr)
#      Nb_avg2 = Nb_avg ** 2
#      delta_Nb = Nb2_avg - Nb_avg2
#      return Nb_avg, delta_Nb
