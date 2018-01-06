import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
from collections import OrderedDict
import pandas as pd

from utils import *


class PrincipalComponent(object):
    """ Class used to perform principal component analysis on worm
    configuration data.
    """
    def __init__(self, L, blocked_val=None):
        self._L = L
        #  self._LL = int(2*self._L)
        #  self._num_pixels = int(self._LL * self._LL)
        self._temps = []
        self._err = {}
        self._eig_vals = {}
        self._eig_vecs = {}
        self._leading_eig_val = {}
        if blocked_val is None:
            self._config_dir = (
                '../data/configs/{}_lattice/separated_data/'.format(self._L)
            )
        else:
            self._config_dir = (
                '../data/blocked_configs/{}_lattice/double_bonds_{}/'.format(
                    self._L, blocked_val
                )
            )
        self._PCA()
        self._leading_eig_val_avg = self.average_data()
        self._leading_eig_val_err = list(self._err.values())

    def _get_data(self, _file=None):
        """ Read in configuration data from file. """
        #  file_path = self._config_dir + file
        try:
            data = pd.read_csv(_file, header=None, engine='c',
                               delim_whitespace=True).values
            configs = data[:, 1:]
        except IOError:
            raise "Unable to read from: {}".format(_file)
        return configs

    def reset_data(self):
        pass

    def calc_PCA(self, data, num_components):
        """ Perform PCA on data to calculate eigenvalue/vector pairs.

        Args:
            data (array-like)
            num_components (int): Number of principal components to keep. 

        Returns:
            self._eig_vals (np.array): Array containing first num_components
                eigenvalues, sorted in decreasing order.
            eig_vecs (np.array): Array containing first num_components
            eigenvectors.
        """
        svd = TruncatedSVD(n_components=num_components)
        svd.fit(data)
        eig_vals = svd.explained_variance_,
        eig_vecs = svd.components_
        return eig_vals, eig_vecs

    def _calc_leading_eigenvalue(self, data, num_components=1):
        """ Calculate leading eigenvalue from calc_PCA method. """
        eig_pairs = self.calc_PCA(data, num_components)
        leading_eig_val = eig_pairs[0][0]
        return leading_eig_val

    def _PCA(self, num_components=1):
        """ Perform PCA analysis file by file for file in self._config_dir to
        prevent memory overflow. 
        """
        _files = os.listdir(self._config_dir)
        files = sorted([
            self._config_dir + i for i in _files if i.endswith('.txt')
        ])
        #  import pdb
        #  pdb.set_trace()
        num_blocks = 10
        for _file in files:
            print("Reading data from: {}".format(_file))
            x = self._get_data(_file)
            num_samples, num_features = x.shape
            if num_samples < num_features:
                continue
            else:
                key = _file.split('_')[-1].rstrip('.txt').rstrip('0')
                #  assert t == float(key)
                self._temps.append(float(key))
                eig_pairs = self.calc_PCA(x, num_components)
                self._eig_vals[key] = eig_pairs[0]
                self._eig_vecs[key] = eig_pairs[1]
                self._leading_eig_val[key] = eig_pairs[0][0]
                x_blocks = block_resampling(x, num_blocks)
                blocked_vals = []
                for block in x_blocks:
                    blocked_vals.append(self._calc_leading_eigenvalue(block))
                self._err[key] = jackknife_err(y_i=blocked_vals,
                                               y_full=eig_pairs[0][0],
                                               num_blocks=num_blocks)
        
        self._eig_vals = OrderedDict(sorted(self._eig_vals.items()))
        self._leading_eig_val = OrderedDict(
            sorted(self._leading_eig_val.items())
        )

    def average_data(self):
        """ Average leading eigenvalue data and use jackknife resampling for
        error bars.
        """
        leading_eig_val_avg = []
        #leading_eig_val_err = []
        for key, val in self._leading_eig_val.items():
            leading_eig_val_avg.append(np.mean(val))
            #  leading_eig_val_err.append(
            #      jackknife_stats(np.array(val), np.std)[2]
            #  )
        return leading_eig_val_avg#, leading_eig_val_err

