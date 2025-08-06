#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.constants as constants
from unit_conversion import eV_to_au, angstrom_to_bohr, lam_to_omega, nm_to_au, fs_to_au, au_to_fs, au_to_Vpm, Vpm_to_au

class BandStructure:

    def __init__(self, Ec, Ev, tc, tv, a):
        """give inputs in eV, angstrom"""
        self.Ec = eV_to_au(Ec)
        self.Ev = eV_to_au(Ev)
        self.tc = eV_to_au(tc)
        self.tv = eV_to_au(tv)
        self.a = angstrom_to_bohr(a)
    
    
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
        """t-end in fs"""
        t_end = fs_to_au(t_end)
        self.time = np.linspace(0, t_end, n_steps)
    
    def make_step(self):
        pass

    def define_pulse(self, sigma, lam, t_start, Enull):
        """lam in nm, sigma in fs, Enull in V/m"""
        t_start = fs_to_au(t_start)
        sigma = fs_to_au(sigma)
        lam = nm_to_au(lam)
        omega = lam_to_omega(lam) 
        Enull = Vpm_to_au(Enull)
        self.E_field = Enull * np.sin(omega * self.time) * np.exp(-(self.time - t_start)**2 / (2 * sigma**2) )

    def define_system(self, num_k, a):
        """a in angstrom"""
        a = angstrom_to_bohr(a)
        self.k_list = np.linspace(-np.pi/a, np.pi/a, num_k)
        self.dens_mat = np.zeros((len(self.time), num_k, 2, 2))
        self.dens_mat[0, :,0 ,0] = 1 # fully populate conduction band 


    def plot_pulse(self):
        fig, ax = plt.subplots()
        time = au_to_fs(self.time)
        E = au_to_Vpm(self.E_field)
        ax.plot(time, E)
        ax.set_xlabel("t/fs")
        ax.set_ylabel(r"E/$Vm^{-1}$")
        plt.show()

class DensityMatrix:

    def __init__(self, k_list, a, time):
        self.k_list = k_list
        self.a = a
        self.time = time


if __name__ =="__main__":
    ZnO = BandStructure(Ec=4, Ev=-3, tc=-1.5, tv=0.5, a=2) #5.16
    # ZnO.plot_bands(num_k=40)
    sim = Simulation(t_end=30, n_steps=1000)
    sim.define_pulse(sigma=3, lam=774, t_start=11, Enull=1)
    sim.define_system(num_k=100, a=9.8) 
    sim.plot_pulse()