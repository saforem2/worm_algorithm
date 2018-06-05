import os
import sys
import argparse
import getopt
import shutil
import numpy as np
import pandas as pd
from utils import *
from collections import OrderedDict

class CountBonds(object):
    """ Class to obtain statistics about the average number of bonds <Nb> and
    the variance in the average number of bonds, <Nb^2> - <Nb>^2

    Args:
        _L (int):
            Linear dimension of lattice.
        _block_val (int):
            Either 0, 1, or 2, specifying the type of blocking scheme used to
            generate the data.
        _num_blocks (int):
            Number of blocks to be used for block bootstrap error analysis.
        _data_dir : str
            Directory containing configuration data to be analyzed.
        _save_dir : (str)
            Directory where resulting bond_statistics data is to be written to.
        _save : bool
            Whether or not to save the bond_statistics data.
        _verbose : bool
            Whether or not to display information as the analysis is being
            performed.
    """
    def __init__(self, L, block_val=None, num_blocks=10,
                 data_dir=None, save_dir=None, data_file=None,
                 save=False, load=False, verbose=False):
        self._L = L
        self._block_val = block_val
        self._num_blocks = num_blocks
        self._verbose = verbose
        #  if block_val is None:
        self._width = 2 * L
        if data_dir is None:
            self._data_dir = (
                '../data/configs/{}_lattice/separated_data/'.format(L)
            )
        else:
            self._data_dir = data_dir
        if save_dir is None:
            self._save_dir = '../data/bond_stats/{}_lattice/'.format(L)
        else:
            self._save_dir = save_dir
        #  else:
        #      self._width = L
        #      if data_dir is None:
        #          self._data_dir = ('../data/blocked_configs/'
        #                            + '{}_lattice/double_bonds_{}/'.format(
        #                                L, block_val))
        #      else:
        #          self._data_dir = data_dir
        #      if save_dir is None:
        #          self._save_dir = (
        #              '../data/bond_stats/{}_lattice/double_bonds_{}/'.format(
        #                  L, block_val
        #              )
        #          )
        #      else:
        #          self._save_dir = save_dir
        if not os.path.exists(self._save_dir):  # ensure save_dir exists
            os.makedirs(self._save_dir)         # create directory if not

        config_files = os.listdir(self._data_dir)
        self._config_files = sorted([
            self._data_dir + i for i in config_files if i.endswith('.txt')
        ])
        temp_strings = [
            i.split('_')[-1].rstrip('.txt') for i in self._config_files
        ]
        self._temp_strings = [i.rstrip('0') for i in temp_strings]
        self.bond_stats = {}
        if not load:
            self.count_bonds()
            self._save()
        else:
            self._load(data_file)
        #  if save:
        #      self._save()

    def _get_configs(self, _file):
        """ Load worm configuration(s) from .txt file.

        Args:
            file (str):
                Path to file containing configuration data.

        Returns:
            config (np.array):
                Array containing worm configurations with one configuration per
                line.
        """
        try:
            return pd.read_csv(_file, header=None, engine='c',
                               delim_whitespace=True, index_col=0).values
        except IOError:
            print("Unable to read from: {}".format(_file))
            raise

    def _count_bonds(self, data):
        #  bc_arr = []
        w = self._width
        bond_idxs = [
            (i, j) for i in range(w) for j in range(w) if (i + j) % 2 == 1
        ]
        #  x = data.reshape(-1, w, w)
        bc_arr = np.array([
            np.sum([config[i] for i in bond_idxs]) for config in
            data.reshape(-1, w, w)
        ])
        #  bc_arr = bc_arr[np.where(bc_arr != 0)]
        bc2_arr = bc_arr ** 2
        Nb_avg = np.mean(bc_arr)
        Nb2_avg = np.mean(bc2_arr)
        Nb_avg2 = Nb_avg ** 2
        delta_Nb = Nb2_avg - Nb_avg2

        #  data_reshaped = data.reshape(-1, w, w)
        #  bc_arr = np.array([np.sum(data_reshaped[i, bond_idxs]) for i in
        #                    range(data_reshaped.shape[0])])
        #  for config in data:
        #      _config = config.reshape(w, w)
        #      _config_bonds = [_config[i] for i in bond_idxs]
        #      bc_arr.append(np.sum(_config_bonds))
        #  bc_arr = np.array(bc_arr)
        #  bc2_arr = bc_arr ** 2
        #  Nb_avg = np.mean(bc_arr)
        #  Nb2_avg = np.mean(bc2_arr)
        #  Nb_avg2 = Nb_avg ** 2
        #  delta_Nb = Nb2_avg - Nb_avg2
        #
        return Nb_avg, delta_Nb
    
    def _count_bonds_with_err(self, data, num_blocks=10):
        """ Calculate the average number of active bonds (Nb) for the worm
        configuration at temperature T using configuration data obtained from
        the get_config method above. 

        Args:
            configs (array-like):
                Array containing configuration data returned from
                self._load_from_file method.

        Returns:
            bond_counts (dict):
                Dictionary with temperature keys and values given by the
                average number of active bonds, Nb. (Averaged over the number
                of sample configurations at a fixed temperature)
        """
        bond_counts = self._count_bonds(data)

        data_rs = block_resampling(data, num_blocks)
        bond_counts_rs = []
        err = []

        for block in data_rs:
            bond_counts_rs.append(self._count_bonds(block))
        bond_counts_rs = np.array(bond_counts_rs)
        for idx in range(len(bond_counts)):
            _err = jackknife_err(y_i=bond_counts_rs[:, idx],
                                 y_full=bond_counts[idx],
                                 num_blocks=num_blocks)
            err.append(_err)

        return bond_counts, err

    def count_bonds(self):
        """Calculate bond statistics for all configuration data."""
        for idx, config_file in enumerate(self._config_files):
            data = None
            if self._verbose:
                print("Reading in from: {}\n".format(config_file))
            #  key = self._temp_strings[idx]
            key = config_file.split('_')[-1].rstrip('.txt')
            #  key = [i.split('_')[-1].rstrip('.txt') for i in ]
            #  i.split('_')[-1].rstrip('.txt') for i in self._config_files
            data = self._get_configs(config_file)
            val, err = self._count_bonds_with_err(data, self._num_blocks)
            self.bond_stats[key] = [val[0], err[0], val[1], err[1]]
            del(data)

    #  def _save_batch(self, data):
        """Save in progress bond_stats data to .txt file."""
        #  save_file = self._save_dir + f'bond_stats_{self._L}_in_progress.txt'
        #  save_file_copy = save_file + '.bak'
        #  orig_exists = os.path.exists(save_file)
        #  copy exists = os.path.exists(save_file_copy)
        #  if orig exists:
        #      shutil.copy(save_file, save_file_copy)
        #      if copy_exists:
        #          os.remove(save_file)
        #  ordered_bond_stats = OrderedDict(sorted(self.bond_stats.items(),
        #                                          key=lambda t: t[0]))
        #  with open(save_file, 'w') as f:
        #      for key, val in ordered_bond_stats.items():
        #          f.write(f'{key} {val[0]} {val[1]} {val[2]} {val[3]}')



    def _save(self):
        """Save bond_stats data to .txt file."""
        save_file = self._save_dir + 'bond_stats_{}.txt'.format(self._L)
        save_file_copy = save_file + '.bak'
        orig_exists = os.path.exists(save_file)
        copy_exists = os.path.exists(save_file_copy)
        if orig_exists:
            shutil.copy(save_file, save_file_copy)
        if orig_exists and copy_exists:
            os.remove(save_file)
        ordered_bond_stats = OrderedDict(sorted(self.bond_stats.items(),
                                                key=lambda t: t[0]))
        with open(save_file, 'w') as f:
            for key, val in ordered_bond_stats.items():
                f.write('{} {} {} {} {}\n'.format(key,
                                                  val[0],
                                                  val[1],
                                                  val[2],
                                                  val[3]))
    def _load(self, data_file=None):
        """Load previously computed bond_stats data from .txt file."""
        if data_file is None:
            data_file = self._save_dir + 'bond_stats_{}.txt'.format(self._L)
        print(f"Reading from: {data_file}")
        raw_data = pd.read_csv(data_file, engine='c', header=None,
                               delim_whitespace=True).values
        for row in raw_data:
            key = str(row[0])
            self.bond_stats[key] = [row[1], row[2], row[3], row[4]]



def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("-L", "--size", type=int, required=True,
                        help="Linear size of lattice.")
    parser.add_argument("-b" "--block_val", type=int,
                        help=("Either 0, 1, or 2; specifies the type of"
                              "blocking scheme used to generate the data."
                              "(default: None)"))
    parser.add_argument("-n" "--num_blocks", type=int, default=10,
                        help=("Number of blocks to be used for block bootstrap"
                              "error analysis (default: 10)"))
    parser.add_argument("-w", "--write", action="store_false",
                        help=("Flag to save resulting data after analysis."
                              "(default: False)"))
    parser.add_argument("-v", "--verbose", action="store_false",
                        help=("Flag to print display information as the"
                              "analysis is being performed. (default: False)"))
    parser.add_argument("--data_dir", type="str", 
                        help=("Directory containing configuration data to be"
                              "analyzed. (default: None)"))
    parser.add_argument("--save_dir", type="str",
                        help=("Directory where resulting bond_statistics"
                              "should be written to. (default: None)"))
    args = parser.parse_args()

    CountBonds(L=args.size,
               block_val=args.block_val,
               num_blocks=args.num_blocks,
               data_dir=args.data_dir,
               save_dir=args.save_dir,
               write=args.write,
               verbose=args.verbose)

if __name__ == '__main__':
    main(sys.argv)
