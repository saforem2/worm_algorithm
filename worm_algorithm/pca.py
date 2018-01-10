import os
import sys
import argparse
import shutil
import getopt
from sklearn.decomposition import TruncatedSVD
from collections import OrderedDict
import numpy as np
import pandas as pd

from utils import *


class PrincipalComponent(object):
    """ Class used to perform principal component analysis on worm
    configuration data.
    
    Attributes
    ----------
    _L : int
        Linear dimension of lattice.
    _block_val : int
        Either 0, 1, or 2, specifying the type of blocking scheme used to
        generate the data.
    _num_blocks : int
        Number of blocks to be used for block bootstrap error analysis.
    _data_dir : str
        Directory containing configuration data to be analyzed.
    _save_dir : str
        Directory where resulting pca data is to be written to.
    _write : bool
        Whether or not to save the pca data.
    _verbose : bool
        Whether or not to display information as the analysis is being
        performed.
    """
    def __init__(self, L, block_val=None, num_blocks=10, data_dir=None,
                 save_dir=None, write=False, verbose=False):
        self._L = L
        self._block_val = block_val
        self._num_blocks = num_blocks
        #  self._LL = int(2*self._L)
        #  self._num_pixels = int(self._LL * self._LL)
        self._temps = []
        self._err = {}
        self._eig_vals = {}
        self._eig_vecs = {}
        self._leading_eig_val = {}
        if data_dir is None:
            if block_val is None:
                self._data_dir = (
                    '../data/configs/{}_lattice/separated_data/'.format(self._L)
                )
            else:
                self._data_dir = (
                    '../data/blocked_configs/{}_lattice'.format(self._L)
                    + '/double_bonds_{}/'.format(block_val)
                )
        else:
            self._data_dir = data_dir
        if save_dir is None:
            if block_val is None:
                self._save_dir = '../data/pca/{}_lattice/'.format(self._L)
            else:
                self._save_dir = (
                    '../data/pca/{}_lattice/'.format(self._L)
                    + 'double_bonds_{}/'.format(self._block_val)
                )
        else:
            self._save_dir = save_dir
        config_files = os.listdir(self._data_dir)
        self._config_files = [
            self._data_dir + i for i in config_files if i.endswith('.txt')
        ]
        temp_strings = [i.split('_')[-1].rstrip('.txt') for i in config_files]
        self._temp_strings = [i.rstrip('0') for i in temp_strings]


        self._PCA()
        self._leading_eig_val_avg = self.average_data()
        self._leading_eig_val_err = list(self._err.values())
        if write:
            self._save()

    def _get_data(self, _file=None):
        """ Read in configuration data from file. """
        #  file_path = self._data_dir + file
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
        """ Perform PCA analysis file by file for file in self._data_dir to
        prevent memory overflow. 
        """
        #  _files = os.listdir(self._data_dir)
        #  files = sorted([
        #      self._data_dir + i for i in _files if i.endswith('.txt')
        #  ])
        #  import pdb
        #  pdb.set_trace()
        #  num_blocks = 10
        #  for _file in files:
        for _file in self._config_files:
            print("Reading data from: {}".format(_file))
            data = self._get_data(_file)
            num_samples, num_features = data.shape
            if num_samples < num_features:
                continue
            else:
                key = _file.split('_')[-1].rstrip('.txt').rstrip('0')
                #  assert t == float(key)
                self._temps.append(float(key))
                eig_pairs = self.calc_PCA(data, num_components)
                self._eig_vals[key] = eig_pairs[0]
                self._eig_vecs[key] = eig_pairs[1]
                self._leading_eig_val[key] = eig_pairs[0][0]
                data_rs = block_resampling(data, self._num_blocks)
                eig_pairs_rs = []
                for block in data_rs:
                    eig_pairs_rs.append(self._calc_leading_eigenvalue(block))
                self._err[key] = jackknife_err(y_i=eig_pairs_rs,
                                               y_full=eig_pairs[0][0],
                                               num_blocks=self._num_blocks)
        self._eig_vals = OrderedDict(
            self._eig_vals.items(), key=lambda t: t[0]
        )
        self._leading_eig_val = OrderedDict(
            self._leading_eig_val.items(), key=lambda t: t[0]
        )

    def average_data(self):
        """ Average leading eigenvalue data and use jackknife resampling for
        error bars.
        """
        leading_eig_val_avg = []
        #leading_eig_val_err = []
        for val in self._leading_eig_val.values():
            leading_eig_val_avg.append(np.mean(val))
            #  leading_eig_val_err.append(
            #      jackknife_stats(np.array(val), np.std)[2]
            #  t
        return leading_eig_val_avg#, leading_eig_val_err

    def _save(self):
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        save_file = self._save_dir + 'leading_eigenvalue_{}.txt'.format(self._L)
        save_file_copy = save_file + '.bak'
        orig_exists = os.path.exists(save_file)
        copy_exists = os.path.exists(save_file_copy)
        if orig_exists:
            shutil.copy(save_file, save_file_copy)
        if orig_exists and copy_exists:
            os.remove(save_file)
        #  ordered_eig_vals = OrderedDict(sorted(self._leading_eig_val.items(),
        #                                        key=lambda t: t[0]))
        with open(save_file, 'w') as f:
            for key, val in self._eig_vals.items():
                f.write("{} {} {}\n".format(key, val[0][0], self._err[key]))

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("-L", "--size", type=int, required=True,
                        help="Linear size of lattice.")
    parser.add_argument("-b", "--block-val",
                        help=("Either 0, 1, or 2; specifies the type of"
                              "blocking scheme used to generate the data."
                              "(default: None)"))
    parser.add_argument("-n", "--num-blocks", type=int, default=10,
                        help=("Number of blocks to be used for block bootstrap"
                              "error analysis (default: 10)"))
    parser.add_argument("-w", "--write", action="store_false",
                        help=("Flag to save resulting data after analysis."
                              "(default: False)"))
    parser.add_argument("-v", "--verbose", action="store_false",
                        help=("Flag to print display information as the"
                              "analysis is being performed. (default: False)"))
    parser.add_argument("-i", "--input",
                        help=("Input directory containing configuration data"
                              "to be analyzed. (default: None)"))
    parser.add_argument("-o", "--output",
                        help=("Output directory where resulting pca analysis"
                              "should be written to. (default: None)"))
    args = parser.parse_args()

    PrincipalComponent(L=args.size,
                       block_val=args.block_val,
                       num_blocks=args.num_blocks,
                       data_dir=args.input,
                       save_dir=args.output,
                       write=args.write,
                       verbose=args.verbose)

if __name__ == '__main__':
    main(sys.argv)
