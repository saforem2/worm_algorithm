import os
import sys
import shutil
import numpy as np
import pandas as pd

def get_config(file):
    """ Load worm configuration(s) from .txt file.

    Args:
        file (str): Path to file containing configuration data.

    Returns:
        config (np.array): Array containing worm configurations with one
            configuration per line.
    """
    try:
        return pd.read_csv(file, header=None, engine='c',
                           delim_whitespace=True, index_col=0).values
    except IOError:
        raise "Unable to read from: {}".format(file)

def bond_counter(L, blocked_val=None, data_dir=None, save_dir=None,
                 write=False):
    """ Calculate the average number of active bonds (Nb) for the worm
    configuration at temperature T using configuration data obtained from the
    get_config method above. 

    Args:
        L (int): Integer specifying the lattice size.
        blocked_val (int): Integer specifying the blocking scheme (used to
            locate configuration data).

    Returns:
        bond_counts (dict): Dictionary with temperature keys and values given
            by the average number of active bonds, Nb. (Averaged over the
            number of sample configurations at a fixed temperature)
    """
    N = 4 * L * L
    if blocked_val is None:
        width = 2 * L
        data_dir = '../data/configs/{}_lattice/separated_data/'.format(L)
        if save_dir is None:
            save_dir = '../data/bond_counts/{}_lattice/'.format(L)
    else:
        width = L
        data_dir = (
            '../data/blocked_configs/{}_lattice/double_bonds_{}/'.format(
                L, blocked_val
            )
        )
        if save_dir is None:
            save_dir = (
                '../data/bond_counts/{}_lattice/double_bonds_{}/'.format(
                    L, blocked_val
                )
            )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = save_dir + 'num_bonds_{}.txt'.format(L)

    if write:
        save_file_copy = save_file + '.bak'
        if os.path.exists(save_file):
            shutil.copy(save_file, save_file_copy)
        if os.path.exists(save_file) and os.path.exists(save_file_copy):
            os.remove(save_file)

    _config_files = sorted([
        i for i in os.listdir(data_dir) if i.endswith('.txt')
    ])
    _temp_strings = [i.split('_')[-1].rstrip('.txt') for i in _config_files]
    temp_strings = [i.rstrip('0') for i in _temp_strings]

    config_files = [data_dir + i for i in _config_files]

    bond_counts = {}
    for idx, file in enumerate(config_files):
        print("Reading from: {}\n".format(file))
        key = temp_strings[idx]
        config = get_config(file)
        bc_arr = []  # bond counts array
        bond_idxs = [
            (i, j) for i in range(width) for j in range(width) if (i+j)%2 == 1
        ]
        for sample in config:
            #  import pdb
            #  pdb.set_trace()
            _sample = sample.reshape(width, width)
            _sample_bonds = [_sample[i] for i in bond_idxs]
            bc_arr.append(np.sum(_sample_bonds))
        bc_arr = np.array(bc_arr)
        bc2_arr = bc_arr ** 2

        mean_bond_counts = np.mean(bc_arr)
        mean_bond2_counts = np.mean(bc2_arr)
        mean_bond_counts2 = mean_bond_counts ** 2
        fluctuations = mean_bond2_counts - mean_bond_counts2

        if write:
            print("Writing to: {}\n".format(save_file))
            with open(save_file, 'a') as f:
                f.write('{} {} {} {} {}\n'.format(key,
                                                  mean_bond_counts,
                                                  mean_bond2_counts,
                                                  mean_bond_counts2,
                                                  fluctuations))
        bond_counts[key] = [mean_bond_counts,
                            mean_bond2_counts,
                            mean_bond_counts2,
                            fluctuations]
    return bond_counts
