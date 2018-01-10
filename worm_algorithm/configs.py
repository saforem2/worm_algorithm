import os
import shutil
import numpy as np
from sklearn.decomposition import TruncatedSVD
from collections import OrderedDict
import pandas as pd

from utils import *

class Configs(object):
    """ Class for performing analysis of worm configurations.

    Since the data files are so large, it is much more efficient to carry out
    all the necessary calculations while the data is temporarily loaded in
    memory. 

    Attributes
    ----------
    """
    def __init__(self, L, blocked_val=None):
        self._L = L
        self._block_val = blocked_val

        if self._block_val is None:
            self._width = 2 * L
            self._config_dir = (
                '../data/configs/{}_lattice/separated_data/'.format(self._L)
            )

        else:
            self._width = L
            self._config_dir = (
                '../data/blocked_configs/{}_lattice/double_bonds_{}/'.format(
                    self._L, self._block_val
                )
            )

        _config_files = sorted([
            f for f in os.listdir(self._config_dir) if f.endswith('.txt')
        ])
        _temp_strings = [i.split('_')[-1].rstrip('.txt') for i in _config_files]
        self._temp_strings = [i.rstrip('0') for i in _temp_strings]
        self._config_files = [self._config_dir + f for f in _config_files]
        #  self.bond_stats, self.leading_eig_vals = self.analysis()
        #  self.bond_counts = self._count_bonds()
        #  self.analysis(write_bond_counts=True, write_pca=True)

    def _load_from_file(self, _file):
        """ Read in configuration data from file. 

        Parameters
        ----------
        _file : str 
            File containing equilibrium worm configurations. Note that each
            file in self._config_dir contains num_samples (independent)
            configurations, with one configuration per line.

        Returns
        -------
        config : np.array, shape = [num_samples, 4* self._L **2]
            Array of configurations.
        """
        try:
            return pd.read_csv(_file, header=None, engine='c',
                               delim_whitespace=True, index_col=0).values
        except IOError:
            print("Unable to read from: {}".format(file))
            raise

    def _count_bonds(self, data):
        bc_arr = []
        w = self._width
        bond_idxs = [
            (i, j) for i in range(w) for j in range(w) if (i + j) % 2 == 1
        ]

        for config in data:
            _config = config.reshape(w, w)
            _config_bonds = [_config[i] for i in bond_idxs]
            bc_arr.append(np.sum(_config_bonds))
        bc_arr = np.array(bc_arr)
        bc2_arr = bc_arr ** 2

        mean_bond_counts = np.mean(bc_arr)
        mean_bond2_counts = np.mean(bc2_arr)
        mean_bond_counts2 = mean_bond_counts ** 2
        fluctuations = mean_bond2_counts - mean_bond_counts2

        bond_counts = [mean_bond_counts,
                       mean_bond2_counts,
                       mean_bond_counts2,
                       fluctuations]
        return bond_counts


    def count_bonds(self, data, num_blocks=10):
        """ Calculate the average number of active bonds (Nb) for the worm
        configuration at temperature T using configuration data obtained from
        the get_config method above. 

        Parameters
        ----------
        configs : array-like
            Array containing configuration data returned from
            self._load_from_file method.

        Returns
        ------
        bond_counts : dict
            Dictionary with temperature keys and values given by the average
            number of active bonds, Nb. (Averaged over the number of sample
            configurations at a fixed temperature)
        """
        bond_counts = self._count_bonds(data)

        data_rs = block_resampling(data, num_blocks)
        #  bond_counts_rs = np.zeros((num_blocks, len(bond_counts)))
        bond_counts_rs = []
        err = []

        for block in data_rs:
            bond_counts_rs.append(self._count_bonds(block))
        bond_counts_rs = np.array(bond_counts_rs)
        for idx in range(len(bond_counts)):
            _err = jackknife_err(y_i=bond_counts_rs[:,idx],
                                 y_full=bond_counts[idx],
                                 num_blocks=num_blocks)
            err.append(_err)

        return bond_counts, err

    def _calc_PCA(self, data, num_components=1):
        """ Perform PCA on configs data to calculate eigenvalue/vector pairs. 
        
        Parameters
        ----------
        configs : array-like
            Array containing a collection of worm configurations, one
            configuration per line.
        num_components : int
            Number of principal components to keep.

        Returns
        -------
        eig_vals : np.array, shape = [num_components]
            Array containing first num_components eigenvalues, sorted in
            decreasing order.
        eig_vecs : np.array, shape = [num_components, len(configs[0])]
            Array containing  first num_components eigenvectors.
        """
        svd = TruncatedSVD(n_components=num_components)
        svd.fit(data)
        eig_vals = svd.explained_variance_,
        eig_vecs = svd.components_
        return eig_vals, eig_vecs

    def calc_leading_eigenvalue(self, data, num_components=1, num_blocks=10):
        """ Perform PCA analysis on configs, keeping the first num_components
        principal components, and use block_resampling for calculating
        jackknife error bars.  """
        
        num_samples, num_features = data.shape
        if num_samples < num_features:
            return 0
        else:
            eig_vals, eig_vecs = self._calc_PCA(data, num_components)
            leading_eig_val = eig_vals[0]
            data_rs = block_resampling(data, num_blocks)
            leading_eig_val_rs = []
            for block in data_rs:
                _eig_vals, _eig_vecs  = self._calc_PCA(block, num_components)
                leading_eig_val_rs.append(_eig_vals[0])
            err = jackknife_err(y_i=leading_eig_val_rs,
                                y_full=leading_eig_val,
                                num_blocks=num_blocks)
        return leading_eig_val, err

    def _write_bond_counts(self, bond_stats):
        save_dir = '../data/bond_counts/{}_lattice/'.format(self._L)
        if self._block_val is not None:
            save_dir += 'double_bonds_{}/'.format(self._block_val)
        #  if not os.path.exists(save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_file = save_dir + 'num_bonds_{}.txt'.format(self._L)
        if os.path.isfile(save_file):
            save_file_copy = save_file + '.bak'
            if os.path.exists(save_file):
                shutil.copy(save_file, save_file_copy)
            bool1 = os.path.exists(save_file)
            bool2 = os.path.exists(save_file_copy)
            if bool1 and bool2:
                os.remove(save_file)
        print("Saving bond statistics to : {}\n".format(save_file)) 
        with open(save_file, 'a') as _f:
            for key, val in self._bond_stats.items():
                _f.write('{} {} {} {} {}\n'.format(key,
                                                   val[0],
                                                   val[1],
                                                   val[2],
                                                   val[3]))
    
    def _write_pca(self, pca_data):
        save_dir = '../data/pca/{}_lattice/'.format(self._L)
        if self._block_val is not None:
            save_dir += 'double_bonds_{}/'.format(self._block_val)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_file = save_dir + 'leading_eigval_{}.txt'.format(self._L)

        if os.path.isfile(save_file):
            save_file_copy = save_file + '.bak'
            if os.path.exists(save_file):
                shutil.copy(save_file, save_file_copy)
            bool1 = os.path.exists(save_file)
            bool2 = os.path.exists(save_file_copy)
            if bool1 and bool2:
                os.remove(save_file)
        print("Saving pca data to: {}\n".format(save_file))
        pca_err = list(self._err.values())
        with open(save_file, 'a') as _f:
            for i in range(len(self._temps)):
                _f.write("{} {} {}\n".format(self._temps[i],
                                             self._leading_eig_val_avg,
                                             pca_err[i]))

    def analysis(self, write_bond_counts=True, write_pca=True):
        """  Calculate bond statistics and perform principal component analysis
        on all configuration data. """
        self.bond_stats = {}
        self.leading_eig_vals = {}
        for idx, config_file in enumerate(self._config_files):
            print("Reading in from: {}\n".format(config_file))
            key = self._temp_strings[idx]
            data = self._load_from_file(config_file)
            self.bond_stats[key] = self.count_bonds(data, num_blocks=10)
            self.leading_eig_vals[key] = (
                self.calc_leading_eigenvalue(data, num_components=1,
                                             num_blocks=10)
            )
        #  if write_bond_counts:
        #      self._write_bond_counts()

        #  if write_pca:
        #      self._write_pca()
        #  return bond_stats, leading_eig_vals


