#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.constants as constants
from unit_conversion import eV_to_au, angstrom_to_bohr, bohr_to_angstrom, lam_to_omega, nm_to_au, fs_to_au, au_to_fs, au_to_Vpm, Vpm_to_au
from scipy.integrate import solve_ivp

class BandStructure:

    def __init__(self, Ec, Ev, tc, tv, a):
        """give inputs in eV, angstrom"""
        self.Ec = eV_to_au(Ec)
        self.Ev = eV_to_au(Ev)
        self.tc = eV_to_au(tc)
        self.tv = eV_to_au(tv)
        self.a = angstrom_to_bohr(a)
    
    
    def get_H_mat(self, k):
        H = np.zeros((2,2))
        H[0,0] = self.Ec + self.tc * np.cos(k * self.a)
        H[1,1] = self.Ev + self.tv * np.cos(k * self.a)
        return H
    
    def plot_bands(self, num_k): # deprecated
        fig, ax = plt.subplots()
        brillzone = np.linspace(-np.pi/self.a, np.pi/self.a, num_k)
        ax.plot(brillzone, self.get_H_mat(brillzone)[0,0])
        ax.plot(brillzone, self.get_H_mat(brillzone)[1,1])
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

    def define_bands(self, Ec, Ev, tc, tv):
        self.bands = BandStructure(Ec=Ec, Ev=Ev, tc=tc, tv=tv, a=bohr_to_angstrom(self.a))
    
    def define_pulse(self, sigma, lam, t_start, Enull):
        """lam in nm, sigma in fs, Enull in V/m"""
        self.t_start = fs_to_au(t_start)
        self.sigma = fs_to_au(sigma)
        lam = nm_to_au(lam)
        self.omega = lam_to_omega(lam) 
        self.E_null = Vpm_to_au(Enull)
        self.E_field = self.E_null * np.sin(self.omega * self.time) * np.exp(-(self.time - self.t_start)**2 / (2 * self.sigma**2) )

    def get_vector_potential(self):
        f = lambda t, y: gaussian_sine(t, self.omega, self.sigma, self.t_start, self.E_null)
        solution = solve_ivp(f, (self.time[0], self.time[-1]), [0], t_eval=self.time,method='DOP853', 
                             rtol=1e-10, 
                             atol=1e-12
                             )
        self.A_field = solution.y[0]

    def define_system(self, num_k, a):
        """a in angstrom"""
        self.num_k = num_k
        self.a = angstrom_to_bohr(a)
        self.k_list = np.linspace(-np.pi/self.a, np.pi/self.a, num_k, endpoint=False)
        self.mat_init = np.zeros((len(self.k_list), 2, 2))
        self.mat_init[:,1,1] = 1 # fully populate conduction band

    def set_h_null(self, dipole_element):
        h_null = np.zeros((len(self.k_list), 2, 2))
        for i,k in enumerate(self.k_list):
            h_null[i] = self.bands.get_H_mat(k)
        h_null[:,1,0] = dipole_element
        h_null[:,0,1] = dipole_element
        h_null = h_null.flatten()
        self.h_null = h_null

    def commute(self, rho):
        H = np.reshape(self.h_null, (len(self.k_list), 2, 2)) 
        rho = np.reshape(rho, (len(self.k_list), 2, 2)) 
        commutator = np.einsum('ijk, ikl -> ijl', H,rho) - np.einsum('ijk, ikl -> ijl', rho, H)
        # commutator = H @ rho - rho @ H 
        return commutator.flatten()

    def get_rhs(self,t, rho):
        rhs = -1j * (self.commute(rho) + self.E_k_function(t)) * self.get_k_partial(rho) #h_null is constant with time
        return rhs 

    def E_k_function(self, t):
        E = self.E_null * np.sin(self.omega * t) * np.exp(-(t - self.t_start)**2 / (2 * self.sigma**2) )
        E_k = E * self.k_list
        E_k = np.repeat(E_k, 4)
        return E_k

    def get_k_partial(self, rho):
        rho = np.reshape(rho, (len(self.k_list), 2, 2)) 
        xplush = np.roll(rho, shift=-1, axis=0)
        xminush = np.roll(rho, shift=1, axis=0)
        xplustwoh = np.roll(rho, shift=-2, axis=0)
        xminustwoh = np.roll(rho, shift=2, axis=0)
        h = self.k_list[1] - self.k_list[0]
        deriv = (8 * xplush - 8 * xminush + xminustwoh - xplustwoh)/(12 * h)
        return deriv.flatten()

    def integrate(self):
        """needs self.rhs"""
        solution = solve_ivp(self.get_rhs, t_span=(self.time[0], self.time[-1]), y0=self.mat_init.flatten(), t_eval=self.time,
                             method='Radau',
                              #atol=1e-12, rtol=1e-12
                              )
        print(np.shape(solution.y))
        print(solution.status)
        self.solution = np.reshape(solution.y.T, (len(self.time),len(self.k_list), 2,2))

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

    def plot_density_matrix(self, k_index):
        fig, ax = plt.subplots()
        ax.plot(self.time, self.solution[:,k_index,0,0]) 
        plt.show()


def gaussian_sine(t, omega, sigma, t_start, E_null):
    return -E_null * np.sin(omega * t) * np.exp(-(t- t_start)**2 / (2 * sigma**2) )


if __name__ =="__main__":
    sim = Simulation(t_end=30, n_steps=1000)
    sim.define_pulse(sigma=3, lam=774, t_start=11, Enull=1)
    sim.define_system(num_k=100, a=9.8) 
    sim.define_bands(Ec=4, Ev=-3, tc=-1.5, tv=0.5)
    sim.set_h_null(dipole_element=0)
    sim.integrate() 
    sim.plot_density_matrix(k_index=50)
    # mat = sim.set_h_null(dipole_element=0)
    # deriv = sim.get_k_partial(sim.h_null)
    # deriv = np.reshape(deriv, (len(sim.k_list), 2, 2)) 
    # plt.plot(sim.k_list, deriv[:,0,0], label="deriv")
    # H = np.reshape(sim.h_null, (len(sim.k_list), 2,2))
    # plt.plot(sim.k_list, H[:,0,0], label="hamilton")
    # plt.legend()
    # plt.show()
    
    

    

# independently: write circularly polarized light
# solve SBE ode integrate
# equidistant time steps
# nyquist theorem for time step estimation delta e delta t roundabout hbar
"""
write laser as function return 4*k 1D array that multiplies with respective k
write dens-mat as k,4, only initial
write function that
    reshapes flat array into matrix shape
    commutes rho and H_k
    flattens
write function that
    reshapes flat array into matrix shape
    calculates periodic k-derivative
    flattens
"""
"""
possible tests:
    check commutator 0
    leave out field: only constant matrix elements

"""
