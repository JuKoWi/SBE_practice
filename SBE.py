#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.constants as constants
from unit_conversion import eV_to_au, angstrom_to_bohr, lam_to_omega, nm_to_au, fs_to_au, au_to_fs, au_to_Vpm, Vpm_to_au
from scipy.integrate import solve_ivp

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
        self.t_start = fs_to_au(t_start)
        self.sigma = fs_to_au(sigma)
        lam = nm_to_au(lam)
        self.omega = lam_to_omega(lam) 
        self.E_null = Vpm_to_au(Enull)
        self.E_field = self.E_null * np.sin(self.omega * self.time) #* np.exp(-(self.time - self.t_start)**2 / (2 * self.sigma**2) )

    def get_vector_potential(self):
        f = lambda t, y: gaussian_sine(t, self.omega, self.sigma, self.t_start, self.E_null)
        solution = solve_ivp(f, (self.time[0], self.time[-1]), [0], t_eval=self.time, rtol=1e-10, atol=1e-12)
        print(solution.nfev)
        self.A_field = solution.y[0]

    def define_system(self, num_k, a):
        """a in angstrom"""
        self.a = angstrom_to_bohr(a)
        self.k_list = np.linspace(-np.pi/a, np.pi/a, num_k)
        self.dens_mat = DensityMatrix(self.k_list, a, self.time)

    def plot_field_E(self):
        fig, ax = plt.subplots()
        time = au_to_fs(self.time)
        E = au_to_Vpm(self.E_field)
        ax.plot(time, E)
        ax.set_xlabel("t/fs")
        ax.set_ylabel(r"E/$Vm^{-1}$")
        plt.show()
    
    def plot_field_A(self):
        fig, ax = plt.subplots()
        time = au_to_fs(self.time)
        A = au_to_fs(self.A_field)
        A = au_to_Vpm(A)
        ax.plot(time, A)
        ax.set_xlabel("t/fs")
        ax.set_ylabel(r"A/$Vsm^{-1}$")
        plt.show()

class DensityMatrix:

    def __init__(self, k_list, a, time):
        self.k_list = k_list
        self.a = a
        self.time = time
        self.mat = np.zeros((len(time), len(k_list), 2, 2))
        self.mat[0,:,0,0] = 1 # fully populate conduction band

    def get_k_deriv(self):
        deriv = np.roll


def gaussian_sine(t, omega, sigma, t_start, E_null):
    return -E_null * np.sin(omega * t) #* np.exp(-(t- t_start)**2 / (2 * sigma**2) )


if __name__ =="__main__":
    ZnO = BandStructure(Ec=4, Ev=-3, tc=-1.5, tv=0.5, a=2) #5.16
    # ZnO.plot_bands(num_k=40)
    sim = Simulation(t_end=30, n_steps=10000)
    sim.define_pulse(sigma=3, lam=774, t_start=11, Enull=1)
    sim.define_system(num_k=100, a=9.8) 
    sim.get_vector_potential()
    sim.plot_field_E()
    sim.plot_field_A()
    

# vectorpotential from E
# independently: write circularly polarized light
# solve SBE ode integrate
# equidistant time steps
# nyquist theorem for time step estimation delta e delta t roundabout hbar