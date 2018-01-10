import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from collections import OrderedDict
from utils import *

def jackknife_err(y_i, y_full, num_blocks):
    return np.sqrt(
        (num_blocks - 1) * np.sum((y_i - y_full)**2, axis=0) / num_blocks
    )

Tc = 2. / np.log(1 + np.sqrt(2))

class SpecificHeat(object):
    """ SpecificHeat class to calculate average energy and specific heat (with
    errors) using observables data generated from WormSimulation.
    """
    def __init__(self, L):
        self._L = L
        self._specific_heat_dict = {}
        self._specific_heat_arr = []
        self._observables_file = (
            '../data/observables/lattice_{}/observables_{}.txt'.format(
                self._L, self._L
            )
        )
        self._energy_dict = self._load_energies()
        self._energy_temps, self._avg_energy, self._energy_error = (
            self._calc_avg_energies()
        )
        #  tup = self._calc_specific_heat()
        self._spec_heat_temps, self._spec_heat, self._spec_heat_err = (
            self._calc_specific_heat()
        )
        #  self._specific_heat_temps, self._specific_heat, self._specific_heat_err
        #  self._specific_heat_temps, self._specific_heat, self._specific_heat_err = (
        #      self._calc_specific_heat()
        #  )
        #  self.calc_specific_heat()
        #  self._avg_specific_heat = self._calc_avg_specific_heat()
        self._summary = {}

    def _load_energies(self):
        """ Load energies from _observables_file and extract average energy.

        Returns
        -------
        energy_dict : (dict)
            Dictionary with temperature keys and energy values from different
            runs.
        """
        try:
            observables = pd.read_csv(self._observables_file,
                                      delim_whitespace=True,
                                      header=None).values
            temps = [str(i).rstrip('0') for i in observables[:, 0]]
            energy_dict = {}
            for idx, temp in enumerate(temps):
                try:
                    energy = observables[idx, 1]
                    if energy > 0:
                        energy *= -1
                    energy_dict[temp].append(energy)
                except KeyError:
                    energy_dict[temp] = [observables[idx, 1]]
            return OrderedDict(sorted(energy_dict.items(), key=lambda t: t[0]))
        except IOError:
            print("Unable to find {}".format(self._observables_file))
            raise

    def _calc_avg_energies(self):
        """ Calculate the average energy at each temperature from
        self._energy_dict. 

        Args:
            None

        Returns:
            temps (list): List containing temperatures at which the
                average energy was calculated.
            avg_energies (list): Values of the average energies.
            errors (list): Standard deviation of energy measurements.
        """
        self._err_dict = {}
        if self._energy_dict is None:
            raise "No values to average. Exiting."
        else:
            temps = []
            avg_energy = []
            errors = []
            for key, val in self._energy_dict.items():
                temps.append(float(key))
                avg_energy.append(np.mean(val))
                errors.append(np.std(val))
                self._err_dict[key] = np.std(val)

        return temps, avg_energy, errors


    def _calc_specific_heat(self, E_avg=None):
        """ Calculate the average specific heat (Cv = dE/dT) using a finite
        difference method. 

        Args:
            None

        Returns:
            temps (list): List containing temperatures at which the specific
            heat was calculated.
            specific_heat (list): List contaning values of the specific heat.
        """
        specific_heat = {} # specific heat dictionary
        #  energy_arr = np.array(list(self._avg_energy.values()))  # (E, err)
        #  E, E_err = energy_arr[:, 0], energy_arr[:, 1]
        #  T = [float(i) for i in list(self._avg_energy.keys())]  # temp array
        specific_heat_temps = []
        specific_heat_vals = []
        if E_avg is None:
            E = self._avg_energy
        else:
            E = E_avg
        T = self._energy_temps
        for i in range(2, len(E)):
            if i in [len(E) - 1, len(E) - 2]:
                continue
            else:
                d1 = np.abs(E[i + 1] - E[i]) / (T[i + 1] - T[i])
                d2 = np.abs(E[i] - E[i - 1]) / (T[i] - T[i - 1])
                d3 = np.abs(E[i + 1] - E[i - 1]) / (T[i + 1] - T[i - 1])
                d4 = np.abs(E[i + 2] - E[i - 2]) / (T[i + 2] - T[i - 2])
                d5 = np.abs(E[i + 2] - E[i]) / (T[i + 2] - T[i])
                d6 = np.abs(E[i + 2] - E[i - 1]) / (T[i + 2] - T[i - 1])
                diff = (d1 + d2 + d3 + d4 + d5 + d6) / 6
                #  sh.append(diff)
                key = str(T[i]).rstrip('0')
                #  specific_heat_temps.append(T)
                #  specific_heat.append(diff)
                try:
                    specific_heat[key].append(diff)
                except KeyError:
                    specific_heat[key] = [diff]
        specific_heat_temps = [
            float(i) for i in list(specific_heat.keys())
        ]
        specific_heat_vals = [
            float(i[0]) for i in list(specific_heat.values())
        ]
        specific_heat_errors = [
            self._err_dict[key] for key in list(specific_heat.keys())
        ]
        return specific_heat_temps, specific_heat_vals, specific_heat_errors

    def calc_specific_heat(self, num_blocks=20):
        """ Calculate specific heat with errors from energy_dict. """
        E_avg = self._avg_energy
        T = self._energy_temps
        self._specific_heat_dict = {}
        self._specific_heat_arr = []
        keys = [str(t).rstrip('0') for t in T]
        sigma = {}
        for idx, t in enumerate(T):
            if idx in [0, 1, len(T) - 1, len(T) -2]:
                continue
            else:
                E = np.array(self._energy_dict[keys[idx]])
                E_m1 = np.array(self._energy_dict[keys[idx-1]])
                E_m2 = np.array(self._energy_dict[keys[idx-2]])
                E_p1 = np.array(self._energy_dict[keys[idx+1]])
                E_p2 = np.array(self._energy_dict[keys[idx+2]])
                lengths = [len(E), len(E_m1), len(E_m2), len(E_p1), len(E_p2)]
                min_len = min(lengths)
                dt1 = T[idx+1] - T[idx]
                dt2 = T[idx] - T[idx-1]
                dt3 = T[idx+1] - T[idx-1]
                dt4 = T[idx+2] - T[idx-2]
                dt5 = T[idx+2] - T[idx]
                dt6 = T[idx+2] - T[idx-1]
                _spec_heat_arr = []
                for j in range(min_len):
                    d1 = np.abs(E_p1[j] - E[j]) / dt1
                    d2 = np.abs(E[j] - E_m1[j]) / dt2
                    d3 = np.abs(E_p1[j] - E_m1[j]) / dt3
                    d4 = np.abs(E_p2[j] - E_m2[j]) / dt4
                    d5 = np.abs(E_p2[j] - E[j]) / dt5
                    d6 = np.abs(E_p2[j] - E_m1[j]) / dt6
                    diff = (d1 + d2 + d3 + d4 + d5 + d6) / 6
                    _spec_heat_arr.append(diff)
                self._specific_heat_dict[str(t).rstrip('0')] = _spec_heat_arr
            E_avg_rs = []
            E_dict_rs = {}
            _temps = []
            for key, val in self._energy_dict.items():
                E_dict_rs[key] = block_resampling(np.array(val), num_blocks)
                E_avg_rs.append(np.mean(val))
                _temps.append(float(key) + 0.01)
            
            _block_size = (
                len(self._energy_dict['1.']) - len(E_dict_rs['1.'][0])
            )

            _, spec_heat_full = self._calc_specific_heat(E_avg_rs)

            E_avg_dict_rs = {}
            for key, val in E_dict_rs.items():
                E_avg_dict_rs[key] = []
                for i in val:
                    E_avg_dict_rs[key].append(np.mean(i))

            sample_blocks = np.array(list(E_avg_dict_rs.values())).T
            spec_heat_rs =[]
            for block in sample_blocks:
                _, ret_val = self._calc_specific_heat(block)
                spec_heat_rs.append(ret_val)

            spec_heat_rs = np.array(spec_heat_rs)
            self._specific_heat_err = jackknife_err(y_i = spec_heat_rs,
                                                    y_full=spec_heat_full,
                                                    num_blocks=num_blocks)

            for key, val in self._specific_heat_dict.items():
                self._specific_heat_arr.append(np.mean(val)) 
            self._specific_heat_dict = OrderedDict(
                sorted(self._specific_heat_dict.items(), key=lambda t: t[0])
            )
            #  return spec_heat_dict, _err
