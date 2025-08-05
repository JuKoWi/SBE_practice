import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.constants as constants

class BandStructure:

    def __init__(self, Ec, Ev, tc, tv, a):
        """give inputs in eV, angstrom"""
        self.Ec = self._eVtoHartree(Ec)
        self.Ev = self._eVtoHartree(Ev)
        self.tc = self._eVtoHartree(tc)
        self.tv = self._eVtoHartree(tv)
        self.a = self._angstrom_to_bohr(a)
    
    def _eVtoHartree(self, val, reverse=False):
        if reverse:
            new_val = val *constants.physical_constants['Hartree energy in eV'][0]
        else:
            new_val = val / constants.physical_constants['Hartree energy in eV'][0]
        return new_val
    
    def _angstrom_to_bohr(self, val):
        bohr_radius_in_meters = constants.physical_constants['Bohr radius'][0]  # in meters
        angstrom_in_meters = constants.angstrom  # 1 Å = 1e-10 m
        return val * angstrom_in_meters / bohr_radius_in_meters
    
    def _get_H_mat(self, k):
        H = np.zeros((2,2, len(k)))
        H[0,0] = self.Ec + self.tc * np.cos(k * self.a)
        H[1,1] = self.Ev + self.tv * np.cos(k * self.a)
        return H
    
    def plot_bands(self, num_k):
        fig, ax = plt.subplots()
        brillzone = np.linspace(-np.pi/self.a, np.pi/self.a, num_k)
        ax.plot(brillzone, self._get_H_mat(brillzone)[0,0])
        ax.plot(brillzone, self._get_H_mat(brillzone)[1,1])
        ax.set_xlabel(r"k/$a_0^{-1}$")
        ax.set_ylabel(r"E/$E_h$")
        plt.show()

"""objective: get density matrix at times t0 to t
diffential equation formulated for each k coupled by k-gradient
all matrices written in abstract two level basis
start with fully populated ground valence band
use runge kutta to integrate in time
"""

class Simulation:

    def __init__(self, t_end, n_steps):
        """t_end in hbar/Eh """
        self.time = np.linspace(0, t_end, n_steps)
    
    def make_step(self):
        pass

    def define_pulse(self, sigma, lam, t_start):
        """lam in nm, sigma in Eh/hbar"""
        omega = self._angular_frequency_au(lam) 
        self.E_field = np.sin(omega * self.time) * np.exp(-(self.time - t_start)**2 / (2 * sigma**2) )

    def define_system(self, num_k, a):
        """a in a_0"""
        self.k_list = np.linspace(-np.pi/a, np.pi/a, num_k)
        self.dens_mat = np.zeros((len(self.time), num_k, 2, 2))


    def plot_pulse(self):
        fig, ax = plt.subplots()
        ax.plot(self.time, self.E_field)
        ax.set_xlabel(r"t/$\hbar E_h^{-1}$")
        ax.set_ylabel(r"E/$E_h e^{-1} a_0^{-1}$")
        plt.show()



    def _angular_frequency_au(self, wavelength_nm):
        # Constants
        c_au = 137.035999084  # speed of light in atomic units
        bohr_radius_m = constants.physical_constants['Bohr radius'][0]  # in meters

        # Convert wavelength from nm to atomic units
        wavelength_m = wavelength_nm * constants.nano  # nm to meters
        wavelength_au = wavelength_m / bohr_radius_m

        # Compute angular frequency in atomic units
        omega_au = 2 * np.pi * c_au / wavelength_au
        return omega_au





if __name__ =="__main__":
    ZnO = BandStructure(Ec=4, Ev=-3, tc=-1.5, tv=0.5, a=2) #5.16
    # ZnO.plot_bands(num_k=40)
    sim = Simulation(t_end=20*41, n_steps=1000)
    sim.define_pulse(sigma=80, lam=774, t_start=300)
    sim.define_system(num_k=100, a=9.8) 

