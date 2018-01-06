import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from collections import OrderedDict

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
        self._observables_file = (
            '../data/observables/lattice_{}/observables_{}.txt'.format(
                self._L, self._L
            )
        )
        self._energy_dict = self._load_data()
        #  self._avg_energy = self._calc_avg_energies()
        self._energy_temps, self._avg_energy, self._energy_error = (
            self._calc_avg_energies()
        )
        self._specific_heat_temps, self._specific_heat = (
            self._calc_specific_heat()
        )
        #  self._avg_specific_heat = self._calc_avg_specific_heat()
        self._summary = {}

    def _load_data(self):
        """ Load energies from _observables_file and extract average energy.

        Args:
            None

        Returns:
            energy_dict (dict): Dictionary with temperature keys and
            energy values from different runs.
        """
        try:
            observables = pd.read_csv(self._observables_file,
                                      delim_whitespace=True,
                                      header=None).values
            temps = [str(i).rstrip('0') for i in observables[:, 0]]
            energy_dict = {}
            for idx, temp in enumerate(temps):
                try:
                    # 3rd column contains the average energy 
                    energy = observables[idx, 1]
                    if energy > 0:
                        energy *= -1
                    energy_dict[temp].append(energy)
                except KeyError:
                    energy_dict[temp] = [observables[idx, 1]]
            return OrderedDict(sorted(energy_dict.items(), key=lambda t: t[0]))
        except IOError:
            raise "Unable to find {}".format(self._observables_file)

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
        return temps, avg_energy, errors


    def _calc_specific_heat(self):
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
        E = self._avg_energy
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
        specific_heat_temps = [float(i) for i in list(specific_heat.keys())]
        specific_heat_vals = [
            float(i[0]) for i in list(specific_heat.values())
        ]
        return specific_heat_temps, specific_heat_vals
