import random
import time
import os
import sys
import argparse
import getopt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from worm_simulation import WormSimulation


class Bonds(WormSimulation):
    """
    Bonds class used for mapping and storing equilibrium configurations as
    vectors which can be interpreted as two-dimensional greyscale images.
    """
    def __init__(self, L, run=False, num_steps=1E7, verbose=True,
                 T_start=1., T_end=3.5, T_step=0.1, write=True):
        if run:
            WormSimulation.__init__(self, L, run, num_steps, verbose,
                                    T_start, T_end, T_step)
        self._L = L
        self._verbose = verbose
        self._bonds_dir = '../data/bonds/lattice_{}/'.format(self._L)
        self._bond_map_dir = '../data/bond_map/lattice_{}/'.format(self._L)
        self.__map_file = self._bond_map_dir + 'bond_map_{}.txt'.format(L)
        self._map = self._get_map()
        self._dict = self._get_bonds()
        self._mapped_bonds = self._map_bonds()
        self._active_x_bonds, self._active_y_bonds = self._get_active_bonds()
        self._active_bonds = {}
        self._x_bonds = {}
        self._y_bonds = {}
        self._config_data = {}
        #  self.get_map()
        #  self.get_bonds()
        #  self.map_bonds()
        #  self.get_active_bonds()
        self.set_config_data()
        if write:
            self.write_config_data()

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
            for idx, file in enumerate(bond_files):
                bonds = pd.read_csv(file, header=None, engine='c',
                                    delim_whitespace=True).values
                key = temp_strings[idx]
                _dict[key] = bonds
        except OSError:
            raise "Unable to locate bond files. Exiting."
        return _dict

    def _map_bonds(self):
        """ Remap bond_dict. """
        _mapped_bonds = {}
        for key, val in self._dict.items():
            bond_sites = []
            for i in range(len(val)):
                bond_sites.append((self._map[val[i,0]], val[i,1]))
            _mapped_bonds[key] = np.array(bond_sites)
        return _mapped_bonds

    def _get_active_bonds(self):
        """
        Extract and return active bonds on the lattice. 

        Returns:
            _active_bonds (dict): Dictionary with attributes:
                _dict.keys(): Temperature (string)
                _dict.values(): np.array of [(start_site), (end_site), value]

        Note:
            _active_bonds has the same structure as self._mapped_bonds, except
            only contains bonds with are active.
        """
        _active_x_bonds = {}
        _active_y_bonds = {}
        for temp in self._mapped_bonds.keys():
            mapped_bonds = np.array(self._mapped_bonds[temp])
            act_x = []
            act_y = []
            for i in mapped_bonds:
                sites = sorted(i[0])
                start_site = np.array(sites[0], dtype=int)
                end_site = np.array(sites[1], dtype=int)
                diff = end_site - start_site
                if i[1] % 2 == 1:
                    if diff[0] == 0:
                        if abs(start_site[1] - end_site[1]) < self._L - 2:
                            act_y.append(sorted([
                                tuple(start_site), tuple(end_site)
                            ]))
                        elif abs(start_site[1] - end_site[1]) >= self._L - 2:
                            if start_site[1] == 0:
                                act_y.append(sorted([
                                    tuple(start_site), tuple(end_site)
                                ]))
                    if diff[1] == 0:
                        if abs(start_site[0] - end_site[0]) < self._L - 2:
                            act_x.append(sorted([
                                tuple(start_site), tuple(end_site)
                            ]))
                        elif abs(start_site[0] - end_site[0]) >= self._L -2:
                            if start_site[0] == 0:
                                act_x.append(sorted([
                                    tuple(start_site), tuple(end_site)
                                ]))
            _active_x_bonds[temp] = act_x
            _active_y_bonds[temp] = act_y
        return _active_x_bonds, _active_y_bonds

    def _set_config_data(self, T):
        """ Remap bonds data from array with shape [1, 2*L*L] to an array with
        shape [2L, 2L], which can be interpreted as a two-dimensional image of
        pixels.

        Args:
            T (float): temperature
        """
        if type(T) is not str:
            T = str(T)
        xb = self._active_x_bonds[T]  # x_bonds
        yb = self._active_y_bonds[T]  # y_bonds
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
                start_site = 2 * site[0]
                end_site = 2 * site[1]
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
            mapped_arr[site[1][0], site[1][1]] = 0

        for bond in y_mapped_bonds:
            mapped_arr[bond[0], bond[1]] = 1
        for site in y_mapped_sites:
            mapped_arr[site[0][0], site[0][1]] = 1
            mapped_arr[site[1][0], site[1][1]] = 1
        
        return mapped_arr

    def set_config_data(self):
        """ Remap data for all T using _set_config_data. """
        for key in self._dict.keys():
            self._config_data[key] = self._set_config_data(key)

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

        for key, val in self._config_data.items():
            self._config_data[key] = (
                np.array(val).reshape(1, -1).tolist()[0]
            )
            config_file = config_dir
            fn = config_dir + '{}_config_{}.txt'.format(self._L, float(key))
            with open(fn, "a") as f:
                f.write('{} {}\n'.format(
                    float(key), ' '.join([str(i) for i in val.flatten()])
                ))
    
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

        if mode == None:
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
            x_blocks_tup = zip(x_blocks_m, x_blocks_p)
            latt_x = np.arange(0, self._L, 1)
            latt_xb = np.arange(0.5, self._L, 2)
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
            fig = plt.figure()
            pca_data = np.array(self._config_data[str(T)]).reshape(2*self._L,
                                                                   2*self._L).T
            plt.imshow(pca_data, cmap='binary')
            plt.xlim((-0.2, 2*self._L))
            plt.ylim((-0.2, 2*self._L))
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


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", "--size", type=int,
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
    bonds = Bonds(L, run=run, num_steps=num_steps, verbose=verbose,
                  T_start=T_start, T_end=T_end, T_step=T_step, write=write)
    print('done.\n')



if __name__ == '__main__':
    main(sys.argv)

