import numpy as np
import pandas as pd
import os
from operator import xor

def _block_config(config, num_block_steps=1, double_bonds_value=0):
    """Create (iterated) blocked config from un-blocked configu using
    approximate (1 + 1 = 0) blocking scheme.

        Args:
            config (array-like):
                Flattened array of length 4L^2 (= 2L * 2L) of pixels
                representing the original configuration.
                (NOTE: L = linear extent of original lattice)
            num_block_steps (int, default = 1):
                Number of blocking steps to perform. If num_block_steps > 1,
                recursively implement block_config on the returned (blocked)
                configuration.
        Returns:
            blocked_config (np.ndarray):
                Flattened blocked configuration of length L^2 /
                2^(num_block_steps) of pixels representing the blocked
                configuration.
    """
    L = int(np.sqrt(len(config.flatten()))/2)
    if config.shape != (2*L, 2*L):
        config = config.reshape(2*L, 2*L)
    blocked_config = np.zeros((L, L), dtype=int)
    blocked_sites = [(2*i, 2*j) for i in range(L//2) for j in range(L//2)]
    for site in blocked_sites:
        i = site[0]
        j = site[1]
        #  look at the number of active external bonds leaving the block to the
        #  right (ext_x_bonds) and upwards (ext_y_bonds)
        ext_x_bonds = [config[2*i, 2*j+3], config[2*i+2, 2*j+3]]
        ext_y_bonds = [config[2*i+3, 2*j], config[2*i+3, 2*j+2]]
        if double_bonds_value == 0:
            ext_x_bonds_active = xor(ext_x_bonds[0], ext_x_bonds[1])
            ext_y_bonds_active = xor(ext_y_bonds[0], ext_y_bonds[1])
            #  active_site = ext_x_bonds_active or ext_y_bonds_active
        else:
            num_active_x_bonds = sum(ext_x_bonds)
            num_active_y_bonds = sum(ext_y_bonds)
            ext_x_bonds_active = 0
            ext_y_bonds_active = 0
            if num_active_x_bonds > 0:
                if num_active_x_bonds == 2:
                    ext_x_bonds_active = double_bonds_value
                if num_active_x_bonds == 1:
                    ext_x_bonds_active = 1
            if num_active_y_bonds > 0:
                if num_active_y_bonds == 2:
                    ext_y_bonds_active = double_bonds_value
                if num_active_y_bonds == 1:
                    ext_y_bonds_active = 1
        active_site = ext_x_bonds_active or ext_y_bonds_active
        blocked_config[i, j] = active_site
        blocked_config[i, j+1] = ext_x_bonds_active
        blocked_config[i+1, j] = ext_y_bonds_active

        for site in blocked_sites:
            i = site[0]
            j = site[1]
            if blocked_config[i, j-1] or blocked_config[i-1, j]:
                blocked_config[site] = 1
    if double_bonds_value == 0:
        while num_block_steps > 1:
            return _block_config(blocked_config.flatten(),
                                 num_block_steps-1, double_bonds_value=0)

    return blocked_config.flatten()

def block_configs(config_file, double_bonds_value=0, out_dir=None):
    """
    Read in all configs stored in `config_file` and construct array of blocked
    configs.

        Args:
            config_file (str):
                File containing original (unblocked) configs, one per line.
    """
    print("Reading from {}".format(config_file))
    try:
        #  temp = file.splitsx./{]}('/')[-1].split('_')[-1].rstrip('.txt')
        temp = config_file.split('/')[-1].split('_')[-1].rstrip('.txt')
        configs = pd.read_csv(config_file, header=None, engine='c', 
                              delim_whitespace=True, index_col=0).values
        L = int(np.sqrt(len(configs[0].flatten()))/2)
        #  blocked_configs = np.zeros((configs.shape[0], configs.shape[1]/4))
        blocked_configs = []
        #  for idx, config in enumerate(configs):
        for config in configs:
            #  blocked_configs[idx] = _block_config(config)
            blocked_configs.append(_block_config(config))
        #blocked_configs = np.array(blocked_configs)
        L_blocked = int(np.sqrt(len(blocked_configs[0].flatten()))/2)
        if out_dir is None:
            out_dir = (f"../data/blocked_configs/{L}_lattice/"
                       + f"double_bonds_{double_bonds_value}/")
            #  out_dir = ("../data/iterated_blocking/"
            #             + "{}_lattice/blocked_{}".format(L, L_blocked))
        save_blocked_configs(blocked_configs, temp, out_dir=out_dir)
    except IOError:
        print("Unable to read from: {}".format(config_file))
        raise


def save_blocked_configs(configs, temperature, out_dir):
    """Save blocked configurations to text file.

        Args:
            configs (array-like):
                Array containing blocked configurations, with one (flattened)
                configuration per row.
            temperature (string / int):
                Temperature at which configurations were generated. Each
                temperature gets its own file.
            out_dir (string):
                Directory in which to save blocked configurations.
        Returns:
            None
    """
    L = int(np.sqrt(len(configs[0].flatten()))/2)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if out_dir.endswith('/'):
        out_file = (
            out_dir + "{}_blocked_configs_{}.txt".format(L, str(temperature))
        )
    else:
        out_file = (
            out_dir + "/{}_blocked_configs_{}.txt".format(L, str(temperature))
        )
    if os.path.exists(out_file):
        os.rename(out_file, out_file + ".bak")
    print("Saving to: {}\n".format(out_file))
    pd.DataFrame(configs).to_csv(out_file, header=None, sep=' ')
