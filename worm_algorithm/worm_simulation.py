import os
import subprocess
import sys
import argparse

import numpy as np

class WormSimulation(object):
    """ WormSimulation, a python wrapper used to implement the worm algorithm
    on a two-dimensional lattice of Ising spins. This class calls on
    'worm_ising_2d.cpp' for the computationally intensive Monte-Carlo
    simulation, and uses file I/O to gather physically important quantities.
    """
    def __init__(self, L=32, run=True, num_steps=1E7, verbose=True,
                 T_start=1., T_end=3.5, T_step=0.1, T_arr=None, bond_flag=1):
        self._L = L
        self._num_steps = num_steps
        self._verbose = verbose
        self._bond_flag = bond_flag
        self._sim_dir = os.getcwd()
        if T_arr is None:
            self._T_range = np.arange(T_start, T_end, T_step)
        else:
            self._T_range = T_arr
        self._run = run
        #  self._display
        self._observables_dir = '../data/observables/lattice_{}/'.format(L)
        if not os.path.exists(self._observables_dir):
            os.makedirs(self._observables_dir)
        self._config_dir = '../data/bonds/lattice_{}/'.format(self._L)
        if not os.path.exists(self._config_dir):
            os.makedirs(self._config_dir)
        self._num_bonds_dir = '../data/num_bonds/lattice_{}/'.format(L)
        if not os.path.exists(self._num_bonds_dir):
            os.makedirs(self._num_bonds_dir)
        self._bond_map_dir = '../data/bond_map/lattice_{}/'.format(L)
        if not os.path.exists(self._bond_map_dir):
            os.makedirs(self._bond_map_dir)
        if self._run:
            self.make()
            self.remove_old_data()
            self.run()

    def prepare_input(self, T, num_steps):
        """ Create txt file to be read in as parameters for the C++ method. """
        seed = T + np.random.randint(1E4) * np.random.rand()
        setup_dir = '../data/setup/'
        if not os.path.exists(setup_dir):
            os.makedirs(setup_dir)
        try:
            self._input_file = '../data/setup/input.txt'
            with open(self._input_file, 'w') as f:
                f.write("%i %.12f %i %.32f %i\n" % (
                    self._L, T, num_steps, seed, self._bond_flag
                ))
        except IOError:
            raise "Unable to locate input file in {}".format(setup_dir)

    def remove_old_data(self):
        """ Remove configuration data from previous runs. """
        try:
            files_ = os.listdir(self._config_dir)
            for _file in files_:
                if _file.endswith('.txt'):
                    os.remove(self._config_dir + _file)
        except ValueError:
            raise "Directory is already empty. Exiting."

    def make(self):
        """ Call the makefile / compile the C++ method, and create the
        executable. Note that the C++ method completes the simulation for a
        fixed, predefined temperature, read in from 'input.txt', which is
        deleted after the simulation is completed.
        """
        try:
            print('compilation -- start\n')
            os.chdir('./src')
            os.system('make clean')
            os.system('make')
            os.chdir('../')
            print('compilation -- done\n')
            self._prog = './src/worm_ising_2d'
            if not os.path.isfile(self._prog):
                #  curr_dir = os.getcwd()
                raise "ERROR: Unable to find executable file {}".format(
                    self._prog
                )
        except (IOError, OSError):
            raise "Directory structure invalid. Exiting."

    def run(self):
        """
        Main method for running the simulation, and storing variables of
        interest.
        """
        #  _L = []
        #  _T = []
        self._T = []
        self._beta = {}
        self._Z = {}
        self._E = {}
        self._Nb = {}
        self._sim_steps = {}
        results = []
        print('runs -- start\n')
        run_number = 0
        for T_idx, T in enumerate(self._T_range):
            #  num_steps = self._num_steps[T_idx]
            num_steps = self._num_steps
            #  if self._verbose:
            print("Running L = {}, T = {}, num_steps:" " {}".format(
                str(self._L), str(T), num_steps
            ))
            self.prepare_input(T, num_steps)
            process = subprocess.Popen([self._prog])
            process.wait()
            try:
                self._output_file = '../data/setup/output.txt'
                results = np.loadtxt(self._output_file)
                L, T, K, Z_by_num_steps, E_avg, Nb_avg, _num_steps = results
                self._beta[str(T).rstrip('0')] = K
                self._Z[str(T).rstrip('0')] = Z_by_num_steps
                self._E[str(T).rstrip('0')] = E_avg
                self._Nb[str(T).rstrip('0')] = Nb_avg
                self._sim_steps[str(T).rstrip('0')] = _num_steps
                self._T.append(T)
                run_number += 1
            except (IOError, OSError):
                raise "Unable to find {}".format(self._output_file)
        try:
            #  observables_file = (
            #      self._observables_dir
            #      + 'observables_{}.txt'.format(self._L)
            #  )
            observables_header = (
                self._observables_dir + 'observables_header.txt'
            )
            observables_description = (
                self._observables_dir + 'observables_description.txt'
            )

            if not os.path.isfile(observables_header):
                with open(observables_header, 'w') as f:
                    f.write('T E_avg Z_avg Nb_avg step_num')
            if not os.path.isfile(observables_description):
                with open(observables_description, 'w') as f:
                    f.write(
                        "T: Temperature of simulation\n"
                        + "E_avg: Averaged energy of the simulation.\n"
                        + "Z_avg: Number of times head==tail / number of"
                            + "steps\n"
                        + "Nb_avg: Average number of active bonds during"
                            + "simulation.\n"
                        + "step_num: Number of steps performed. """
                    )
            #  if os.path.isfile(observables_file):
            #      with open(observables_file, 'a') as f:
            #          for key in self._E.keys():
            #              f.write('{} {} {} {} {}\n'.format(
            #                  key,
            #                  self._beta[key],
            #                  self._Z[key],
            #                  self._E[key],
            #                  self._Nb[key]
            #              ))
            #  else:
            #      with open(observables_file, 'w') as f:
            #          for key in self._E.keys():
            #              f.write('{} {} {} {} {}\n'.format(
            #                  key,
            #                  self._beta[key],
            #                  self._Z[key],
            #                  self._E[key],
            #                  self._Nb[key]
            #              ))
        except (IOError, OSError):
            raise IOError("Unable to write header/description files.")
    
    def clean(self):
        """ Remove 'input.txt', 'output.txt' files. """
        try:
            os.remove(self._input_file)
            os.remove(self._output_file)
        except OSError:
            raise (
                "Unable to find input/output txt files in:\n "
                + "{}\n {}\n".format(self._input_file, self._output_file)
            )

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", "--size", type=int,
                        help="Define the linear size of the lattice (DEFAULT:"
                             "32)")
    parser.add_argument("-r", "--run_sim", action="store_true",
                        help="Run simulation (bool) (DEFAULT: True)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help=("Display information while MC simulation runs"
                              "(bool) (DEFAULT: True)"))
    parser.add_argument("-p", "--power", type=int,
                        help=("Set the exponent for the number of steps to"
                              "take for MC sim, 10 ** pow (DEFAULT: 6)"))
    parser.add_argument("-T0", "--Tstart", type=float,
                        help = ("Starting temperature for simulation."
                                "(float) (DEFAULT: T_start=1.0)"))
    parser.add_argument("-T1", "--Tend", type=float,
                        help = ("Ending temperature for simulation."
                                "(float) (DEFAULT: T_end=3.5)"))
    parser.add_argument("-t", "--Tstep", type=float,
                        help = ("Temperature step for simulation."
                                "(float) (DEFAULT: T_end=0.1)"))
    args = parser.parse_args()

    L = args.size
    if L is None:
        L = 32

    power = args.power
    if power is None:
        num_steps = 1E7
    run = args.run_sim
    verbose = args.verbose
    T_start = args.Tstart
    if T_start is None:
        T_start = 1.0

    T_end = args.Tend
    if T_end is None:
        T_end = 3.5

    T_step = args.Tstep
    if T_step is None:
        T_step = 0.1

    print("Initializing simulation...\n")
    WormSimulation(L, run=run, num_steps=num_steps, verbose=verbose,
                   T_start=T_start, T_end=T_end, T_step=T_step)
    print('done.\n')


if __name__ == '__main__':
    main(sys.argv)
