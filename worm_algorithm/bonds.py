import os
import sys
import argparse
import getopt
from operator import xor
from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from worm_simulation import WormSimulation


class Bonds(WormSimulation):
    """
    Lattice class used for mapping and storing equilibrium configurations as
    vectors which can be interpreted as two-dimensional greyscale images.

    Attributes:
        L (int):
            Linear size of lattice.
        run (bool):
            Flag to run the simulation (if True)
        num_steps (int):
            Number of MC steps to carry out in simulation.
        decay_steps (bool):
            Flag for decaying num_steps with temperature, since the simulation
            becomes much less efficient at higher T. (NOTE: This flag should
            probably not be enabled, it seems to give bad data.)
        verbose (bool):
            Flag for printing information about the simulation as its being
            run. 
        T_start (float):
            Starting temperature for simulation.
    """
    def __init__(self, L, run=False, num_steps=1E7, decay_steps=False,
                 verbose=True, T_start=1., T_end=3.5, T_step=0.1, T_arr=None,
                 block_configs=False, block_val=0,
                 write=True, write_blocked=False):
        """Initialize Bonds class, which can also be used to run the
        simulation."""
        if T_arr is None:
            WormSimulation.__init__(self, L, run, num_steps, decay_steps,
                                    verbose, T_start, T_end, T_step)
        else:
            WormSimulation.__init__(self, L, run, num_steps, decay_steps,
                                    verbose, T_arr=T_arr)
        self._L = L
        self._num_bonds = 2*self._L*self._L
        self._bonds_dir = '../data/bonds/lattice_{}/'.format(self._L)
        self._bond_map_dir = '../data/bond_map/lattice_{}/'.format(self._L)
        self.__map_file = self._bond_map_dir + 'bond_map_{}.txt'.format(L)
        self._map = self._get_map()
        self._dict = self._get_bonds()
        self._mapped_bonds = self._map_bonds()
        tup = self._get_active_bonds()
        self._active_bonds, self._active_x_bonds, self._active_y_bonds = tup
        self._x_bonds = {}
        self._y_bonds = {}
        self._config_data = self.set_config_data()
        if block_configs:
            self._blocked_config_data = self.block_configs(block_val)
        if write:
            self.write_config_data()
        if write_blocked:
            self.write_blocked_config_data()

    def _get_raw_bonds(self):
        """ Read in raw bond/site data.
         A bond is defined between two sites, e.g.
            start_site <---> end_site
        where each row in self.__map_file is formatted as:
            bond_index start_site_x start_ste_y end_site_x end_site_y
        """
        try:
            self._raw_bonds = pd.read_csv(
                self.__map_file, header=None, engine='c', delim_whitespace=True
            ).values
        except (IOError, OSError):
            raise "ERROR: Unable to find {}".format(self.__map_file)

    def _get_map(self):
        """ Use self.__map_file to create a dictionary mapping bonds to (x,y)
        sites.

        Since the C++ method indexes bonds linearly from 0, ..., 2*L*L, we
        provide the self.__map_file which maps these indices to their
        appropriate (x,y) positions on the 2D lattice.

        Since a bond is defined between two sites, e.g.
            start_site <---> end_site
        each row in self.__map_file is formatted as:
            bond_index start_site_x start_site_y end_site_x end_site_y
        so we construct dictionary self._map formatted as:
            self._map[bond_index] = [
                (start_site_x, start_site_y), (end_site_x, end_site_y)
            ]
        """
        self._get_raw_bonds()
        self._raw_bonds = np.array(self._raw_bonds, dtype=int)
        _map = {}
        for i in range(len(self._raw_bonds)):
            key = self._raw_bonds[i, 0]
            sites = self._raw_bonds[i, 1:]
            start_site = sites[:2]
            end_site = sites[2:]
            _map[key] = [tuple(start_site), tuple(end_site)]
        #########################################
        #      BROKEN BELOW
        #  keys = self._raw_bonds[:, 0]
        #  start_sites = self._raw_bonds[:, 1:3]
        #  end_sites = self._raw_bonds[:, 3:]
        #  for idx, key in enumerate(keys):
        #      try:
        #          _map[key].append([tuple(start_site[idx]),
        #                            tuple(start_site[idx])])
        #      except KeyError:
        #          _map[key] = [tuple(start_site[idx]), tuple(start_site[idx])]
        #########################################

        #  for i in range(len(self._raw_bonds)):
        #      key = self._raw_bonds[i, 0]
        #      sites = self._raw_bonds[i, 1:]
        #      start_site = sites[:2]
        #      end_site = sites[2:]
            #  try:
            #      _map[key].append([tuple(start_site), tuple(end_site)])
            #  except KeyError:
            #  _map[key] = [tuple(start_site), tuple(end_site)]
        return _map

    def _get_bonds(self):
        """ Read in bond data for each temperature.

        Returns:
            _dict (dict): Dictionary containing active bonds and their indices
            for each temperature. _dict has the following attributes:
                _dict.keys(): Temperature (str)
                _dict.values(): [bond_idx, status], where status = 1 if bond is
                active, 0 otherwise.
        """
        _dict = {}
        try:
            bond_files = [
                self._bonds_dir + f for f in os.listdir(self._bonds_dir)
                if f.endswith('.txt')
            ]
            split_files = [i.split('/') for i in bond_files]
            temp_strings = [i[-1].split('_')[-1].rstrip('.txt').rstrip('0') for
                            i in split_files]
            for idx, f in enumerate(bond_files):
                bonds = pd.read_csv(f, header=None, engine='c',
                                    delim_whitespace=True).values
                key = temp_strings[idx]
                num_configs = bonds.shape[0] / self._num_bonds
                split_configs = np.split(bonds, num_configs)
                config_dict = {}
                #for row in bonds[
                for idx, config in enumerate(split_configs):
                    bond_dict = {}
                    for row in config:
                        b_idx = row[0]
                        bond_dict[b_idx] = row[1]
                    config_dict[idx] = bond_dict
                        #  try:
                        #      bond_dict[site].append(row[1])
                        #  except KeyError:
                        #      bond_dict[site] = [row[1]]
                _dict[key] = config_dict
                    #  _dict[key].append(bonds)
                    #  _dict[key] = bonds
        except OSError:
            raise "Unable to locate bond files. Exiting."
        return _dict

    def _map_bonds(self):
        """ Remap bond_dict. """
        _mapped_bonds = {}
        for temp, config_dict in self._dict.items():
            configs_mapped_bonds = {}
            for config, row in config_dict.items():
                bond_sites = []
                for idx, val in row.items():
                    bond_sites.append([self._map[idx], val])
                configs_mapped_bonds[config] = np.array(bond_sites)
            _mapped_bonds[temp] = configs_mapped_bonds

            #  bond_sites = []
            #  for key, val in bond_dict.items():
            #      bond_sites.append([self._map[key], np.array(val)])
            #  _mapped_bonds[temp] = np.array(bond_sites)
        #  for key, val in self._dict.items():
        #      bond_sites = []
        #      for i in range(len(val)):
        #          bond_sites.append((self._map[val[i,0]], val[i,1]))
        #      _mapped_bonds[key] = np.array(bond_sites)
        return _mapped_bonds

    def _get_active_bonds(self):
        """ 
        Split active bonds into x_bonds and y_bonds.

        Returns:
            x_bonds, y_bonds (dict, dict): With temperature (string) keys.
        """
        active_bonds = {}
        for temp, config_dict in self._mapped_bonds.items():
            configs_active_bonds = {}
            for config_idx, bond_arr in config_dict.items():
                configs_active_bonds[config_idx] = []
                for row in bond_arr:
                    if row[1] % 2 == 1:
                        configs_active_bonds[config_idx].append(row[0])
                        active_bonds[temp] = configs_active_bonds
        x_bonds = {}
        y_bonds = {}
        for temp, val in active_bonds.items():
            configs_x_bonds = {}
            configs_y_bonds = {}
            for config_idx, config_bonds in val.items():
                x_sites = []
                y_sites = []
                for site in config_bonds:
                    start_site = np.array(site[0])
                    end_site = np.array(site[1])
                    diff = abs(start_site - end_site)
                    if diff[0] in [1, self._L - 1]:
                        x_sites.append(sorted(tuple(site)))
                    elif diff[1] in [1, self._L - 1]:
                        y_sites.append(sorted(tuple(site)))
                configs_x_bonds[config_idx] = (
                    np.array(x_sites, dtype=int).tolist()
                )
                configs_y_bonds[config_idx] = (
                    np.array(y_sites, dtype=int).tolist()
                )
            x_bonds[temp] = configs_x_bonds
            y_bonds[temp] = configs_y_bonds
        return active_bonds, x_bonds, y_bonds

    def _set_config_data(self, T, config_idx):
        """ Remap bonds data from array with shape [1, 2*L*L] to an array with
        shape [2L, 2L], which can be interpreted as a two-dimensional image of
        pixels.

        Args:
            T (float): temperature
        """
        #  if type(T) is not str:
        if not isinstance(T, str):
            T = str(T)
        #  import pdb
        #  pdb.set_trace()
        xb = self._active_x_bonds[T][config_idx]  # x_bonds
        yb = self._active_y_bonds[T][config_idx]  # y_bonds
        xbl = [[list(i[0]), list(i[1])] for i in xb]  # x_bonds_list
        ybl = [[list(i[0]), list(i[1])] for i in yb]  # y_bonds_list
        #  for i in xb:
        #      sites_list = [list(i[0]), list(i[1])]
        #      xbl.append(sites_list)
        #  for i in yb:
        #      sites_list = [list(i[0]), list(i[1])]
        #      ybl.append(sites_list)
        mapped_arr = np.zeros((2*self._L, 2*self._L), dtype=int)

        x_mapped_sites = []
        x_mapped_bonds = []
        for bond in xbl:
            sites = []
            x_diff = np.abs(bond[1][0] - bond[0][0])
            if (x_diff == self._L - 1):
                y_mapped = bond[0][1] + bond[1][1]
                xR_mapped = bond[1][0] + bond[1][0] + 1
                x_mapped_bonds.append([xR_mapped, y_mapped])
            else:
                x_mapped = bond[0][0] + bond[1][0]
                y_mapped = bond[0][1] + bond[1][1]
                x_mapped_bonds.append([x_mapped, y_mapped])
            for site in bond:
                start_site = site[0] + site[0]
                end_site = site[1] + site[1]
                sites.append([start_site, end_site])
            x_mapped_sites.append(sites)
        x_mapped_sites = np.array(x_mapped_sites, dtype=int)
        x_mapped_bonds = np.array(x_mapped_bonds, dtype=int)

        y_mapped_sites = []
        y_mapped_bonds = []
        for bond in ybl:
            sites = []
            y_diff = np.abs(bond[1][1] - bond[0][1])
            if (y_diff == self._L - 1):
                x_mapped = bond[0][0] + bond[1][0]
                yU_mapped = bond[1][1] +  bond[1][1] + 1
                y_mapped_bonds.append([x_mapped, yU_mapped])
            else:
                x_mapped = bond[0][0] + bond[1][0]
                y_mapped = bond[0][1] + bond[1][1]
                y_mapped_bonds.append([x_mapped, y_mapped])
            for site in bond:
                start_site = site[0] + site[0]
                end_site = site[1] + site[1]
                sites.append([start_site, end_site])
            y_mapped_sites.append(sites)
        y_mapped_sites = np.array(y_mapped_sites, dtype=int)
        y_mapped_bonds = np.array(y_mapped_bonds, dtype=int)

        for bond in x_mapped_bonds:
            mapped_arr[bond[0], bond[1]] = 1
        for site in x_mapped_sites:
            mapped_arr[site[0][0], site[0][1]] = 1
            mapped_arr[site[1][0], site[1][1]] = 1

        for bond in y_mapped_bonds:
            mapped_arr[bond[0], bond[1]] = 1
        for site in y_mapped_sites:
            mapped_arr[site[0][0], site[0][1]] = 1
            mapped_arr[site[1][0], site[1][1]] = 1
        return mapped_arr

    def set_config_data(self):
        """ Remap data for all T using _set_config_data. """
        config_data = {}
        for temp in self._dict.keys():
            _config_data = {}
            for config_idx in self._dict[temp].keys():
                _config_data[config_idx] = self._set_config_data(temp,
                                                                 config_idx)
            config_data[temp] = _config_data
            #config_data[key] = self._set_config_data(key)
        return OrderedDict(sorted(config_data.items(), key=lambda t: t[0]))
    
    def _block_config_T(self, T, config_idx, block_val):
        """ Block individual configuration (at temp T).

        By `block` we mean implement a renormalization group `coarse graining`
        transformation to the worm configuration, where groups of four sites
        are blocked into a single site in the blocked configuration. To deal
        with the `double bonds` (defined as the case when two bonds exit a
        given block in the original lattice) we include a parameter `block_val`
        that assigns a weight to these double bonds when constructing the
        blocked configuration. 

        If block_val is 0, we call this method `approximate` blocking, which
        has the advantage of maintaining closed-path configurations in the
        blocked configuration, but often loses information about the bonds in
        the original configuration. Additionally, this method is able to be
        performed iteratively. 

        Conversely, if block_val is not 0, we no longer maintain the
        closed-loop configuration under blocking, but we more accurately
        represent the original configuration. 

        Args:
            T (float or str): 
                If T is a float, re-cast it as a string (removing trailing
                zeros) to be used as a key in self._config_data dictionary
                which stores the equilibrium configurations resulting from the
                worm algorithm.
            block_val (int): 
                Value given to double bonds when constructing the blocked
                configuration. 

        Returns:
            blocked_config (aray-like): 
                If self._config_data['T'] is of shape [2L, 2L], blocked_config
                is an array of shape [L, L], representing the configuration
                after having performed a single renormalization group
                `blocking` iteration.
        """
        #  if type(T) is not str:
        if not isinstance(T, str):
            T = str(T).rstrip('0')
        config = np.array(self._config_data[T][config_idx]).reshape(
            2*self._L, 2*self._L
        )
        blocked_config = np.zeros((self._L, self._L), dtype=int)
        blocked_sites = [
            (2*i, 2*j) for i in range(self._L//2) for j in range(self._L//2)
        ]
        for site in blocked_sites:
            i = site[0]
            j = site[1]
            ext_x_bonds = [config[2*i, 2*j+3], config[2*i+2, 2*j+3]]
            ext_y_bonds = [config[2*i+3, 2*j], config[2*i+3, 2*j+2]]
            ext_x_bonds_active = xor(ext_x_bonds[0], ext_x_bonds[1])
            ext_y_bonds_active = xor(ext_y_bonds[0], ext_y_bonds[1])
            active_site = ext_x_bonds_active or ext_y_bonds_active
            blocked_config[i, j] = active_site
            blocked_config[i, j+1] = ext_x_bonds_active
            blocked_config[i+1, j] = ext_y_bonds_active
            if block_val != 0:
                if ext_x_bonds == [1, 1]:
                    blocked_config[i, j] = block_val
                    blocked_config[i, j+1] = block_val
                if ext_y_bonds == [1, 1]:
                    blocked_config[i, j] = block_val
                    blocked_config[i+1, j] = block_val

        for site in blocked_sites:
            i = site[0]
            j = site[1]
            x_site = blocked_config[i, j-1]
            y_site = blocked_config[i-1, j]
            if x_site == 1 or y_site == 1:
                blocked_config[site] = 1
            if block_val != 0:
                if x_site == block_val or y_site == block_val:
                    blocked_config[site] = block_val
        return blocked_config
    
    def block_configs(self, block_val=0):
        blocked_configs = {}
        blocked_configs[block_val] = {}
        #  for temp, val in self._config_data.items():
        for temp in self._config_data.keys():
            _blocked_configs = {}
            _blocked_configs[block_val] = {}
            for config_idx in self._config_data[temp].keys():
                _blocked_configs[block_val][config_idx] = (
                    self._block_config_T(temp, config_idx, block_val)
                )
            blocked_configs[block_val][temp] = _blocked_configs[block_val]
            #  blocked_configs[block_val][key] = self._block_config_T(key,
            #                                                         block_val)
        return blocked_configs

    def write_config_data(self, data_dir=None):
        """ Save configuration data to be used for later analysis. 

        Args:
            data_dir (str): Path to directory for saving configuration data. 

        Returns:
            None
        """
        if data_dir is None:
            config_dir = (
                '../data/configs/{}_lattice/separated_data/'.format(self._L)
            )
        else:
            config_dir = data_dir

        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        for temp in self._config_data.keys():
            tf = float(temp)
            fn = config_dir + '{}_config_{}.txt'.format(self._L, tf)
            print("Saving configs to: {}".format(fn))
            for config in self._config_data[temp].values():
                #  self._config_data[key] = (
                #      np.array(val).reshape(1, -1).tolist()[0]
                #  )
                #  config_file = config_dir
                with open(fn, "a") as f:
                    f.write('{} {}\n'.format(
                        tf, ' '.join([str(i) for i in config.flatten()])
                    ))

    def write_blocked_config_data(self, data_dir=None, blocked_configs=None):
        """ Save blocked configuration data to be used for later analysis.

        Args:
            data_dir (str): Path to directory for saving blocked configuration
                data.
            block_val (int): Value for double bonds, same as key of
                self._blocked_config_data.

        Returns:
            None
        """
        if blocked_configs is None:
            configs = self._blocked_config_data
            block_val = list(configs.keys())[0]
        else:
            configs = blocked_configs
            block_val = list(configs.keys())[0]
        if data_dir is None:
            config_dir = (
                '../data/blocked_configs/{}_lattice/double_bonds_{}/'.format(
                    self._L, block_val
                )
            )
        else:
            config_dir = data_dir
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        try:
            for temp in configs[block_val].keys():
                tf = float(temp)
                config_file = config_dir + '{}_config_{}.txt'.format(self._L,
                                                                     tf)
                print("Saving blocked configs to: {}".format(config_file))
                for c in self._blocked_config_data[block_val][temp].values():
                    with open(config_file, "a") as f:
                        f.write('{} {}\n'.format(
                            tf, ' '.join([str(j) for j in c.flatten()])
                        ))
        except KeyError:
            raise ("Block val of blocked configs is not the same as "
                   "{}. Exiting. ".format(block_val))
    
    def plot_config(self, T, mode=None, show=True, save=False, save_dir=None):
        """ Generate plot of worm configuration.

        Args:
            T (float): Temperature specifying what data to use for plotting.
            mode (str): If mode is None, generate 'pretty' plot for
                visualization purposes. If mode is 'pca', create plot using
                binary-valued pixels, extracted from self._config_data
            show (boolean): Display the plot / image
            save (boolean): Save the plot / image
        """
        if T is not str:
            T = str(T).rstrip('0')

        if show:
            plt.clf()
            plt.close('all')

        if mode is None:
            plt.clf()
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(1,1,1)
            fig.add_axes(ax)
            major_ticks = np.arange(0, self._L, 10)
            ax.set_xticks(major_ticks)
            ax.set_yticks(major_ticks)
            #  title_str =  'Worm configuration, T = ' + T
            #  plt.title(title_str)
            plt.xlim((-0.4, self._L-1+0.4))
            plt.ylim((-0.4, self._L-1+0.4))
            mapped_bonds = np.array(self._mapped_bonds[T])
            x_blocks_p = []
            x_blocks_m = []

            for i in range(0, self._L, 2):
                x_blocks_m.append(i - 0.2)
            for i in range(1, self._L, 2):
                x_blocks_p.append(i + 0.2)
            #  x_blocks_tup = zip(x_blocks_m, x_blocks_p)
            latt_x = np.arange(0, self._L, 1)
            #  latt_xb = np.arange(0.5, self._L, 2)
            for i in range(self._L):
                for j in range(self._L):
                    plt.plot(latt_x[i], latt_x[j], color='k',
                             marker='s', markersize=2)
                    plt.plot(i, j, color='k', marker='s', markersize=1)
            for i in mapped_bonds:
                sites = sorted(i[0])
                start_site = np.array(sites[0], dtype=int)
                end_site = np.array(sites[1], dtype=int)
                diff = end_site - start_site
                if i[1] % 2 == 1:
                    if diff[0] == 0:
                        if abs(start_site[1] - end_site[1]) < self._L - 2:
                            plt.vlines(start_site[0], start_site[1],
                                       end_site[1], linewidth=3,
                                       color='k', linestyle='-',
                                       zorder=15)
                        elif abs(start_site[1] - end_site[1]) >= self._L - 2:
                            if start_site[1] == 0:
                                plt.vlines(start_site[0], start_site[1], -0.5,
                                          linewidth=3, color='k',
                                           linestyle='-', zorder=15)
                                plt.vlines(start_site[0], end_site[1],
                                           end_site[1] + 0.5, linewidth=3,
                                           color='k', linestyle='-',
                                           zorder=15)
                    if diff[1] == 0:
                        if abs(start_site[0] - end_site[0]) < self._L - 2:
                            plt.hlines(start_site[1], start_site[0],
                                       end_site[0], linewidth=3,
                                       color='k', linestyle='-',
                                       zorder=15)
                        elif abs(start_site[0] - end_site[0]) >= self._L - 2:
                            if start_site[0] == 0:
                                plt.hlines(start_site[1], start_site[0], -0.5,
                                          linewidth=3, color='k',
                                           linestyle='-', zorder=15)
                                plt.hlines(start_site[1], end_site[0],
                                           end_site[0] + 0.5, linewidth=3,
                                           color='k', linestyle='-',
                                           zorder=15)
                    #  plt.plot(latt_x[i], latt_x[j], color='')
            if save:
                fig_title = 'worm_config_{}_.png'.format(str(T).rstrip('.0'))
                if save_dir is None:
                    save_dir = '../plots/configs/lattice_{}/'.format(self._L)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                print("Saving worm configuration to: {}".format(save_dir +
                                                                fig_title))
                plt.savefig(save_dir + fig_title, dpi=400, bbox_inches='tight')
            if show:
                plt.show()
        elif mode=='pca':
            plt.clf()
            fig, ax = plt.subplots()
            major_ticks = np.arange(0, self._L, 2)
            ax.set_xticks(major_ticks)
            ax.set_yticks(major_ticks)
            pca_data = np.array(self._config_data[str(T)]).reshape(2*self._L,
                                                                   2*self._L).T
            plt.imshow(pca_data, cmap='binary')
            plt.xlim((0, 2*self._L-1))
            plt.ylim((0, 2*self._L-1))
            if save:
                if save_dir is None:
                    save_dir = (
                        '../plots/configs/lattice_{}/images/'.format(self._L)
                    )
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                output_file = (
                    save_dir + 'pca_worm_config_{}_.png'.format(
                        str(T).rstrip('.0')
                    )
                )
                print("Saving worm configuration to {}".format(output_file))
                plt.savefig(output_file, dpi=400, bbox_inches='tight')
            if show:
                plt.clf()
                plt.show()

        return fig


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", "--size", type=int, required='True',
                        help="Define the linear size of the lattice (DEFAULT:"
                             "16)")
    parser.add_argument("-r", "--run_sim", action="store_true",
                        help="Run simulation (bool) (DEFAULT: True)")
    #  parser.add_argument("-p", "--plot", action="store_true",
    #                      help="Create plots for each run (bool)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help=("Display information while MC simulation runs"
                              "(bool) (DEFAULT: True)"))
    parser.add_argument("-p", "--power", type=int,
                        help=("Set the exponent for the number of steps to"
                              "take for MC sim, 10 ** pow (DEFAULT: 6)"))
    parser.add_argument("-w", "--write", action="store_true",
                        help=("Write worm config to data file. (bool)"
                              "(DEFAULT: True)"))
    parser.add_argument("-T0", "--Tstart", type=float,
                        help = ("Starting temperature for simulation."
                                "(float) (DEFAULT: T_start=1.4)"))
    parser.add_argument("-T1", "--Tend", type=float,
                        help = ("Ending temperature for simulation."
                                "(float) (DEFAULT: T_end=3.5)"))
    parser.add_argument("-t", "--Tstep", type=float,
                        help = ("Temperature step for simulation."
                                "(float) (DEFAULT: T_end=0.1)"))
    args = parser.parse_args()

    L = args.size

    power = args.power
    if power is None:
        num_steps = 1E7
    else:
        num_steps = 10**power

    run = args.run_sim
    verbose = args.verbose
    write = args.write
    T_start = args.Tstart
    if T_start is None:
        T_start = 1.0

    T_end = args.Tend
    if T_end is None:
        T_end = 3.5

    T_step = args.Tstep
    if T_step is None:
        T_step = 0.1

    print("Initializing bonds..."),
    Bonds(L, run=run, num_steps=num_steps, verbose=verbose,
          T_start=T_start, T_end=T_end, T_step=T_step, write=write)
    print('done.\n')



if __name__ == '__main__':
    main(sys.argv[1:])
