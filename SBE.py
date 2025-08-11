#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.constants as constants
from unit_conversion import eV_to_au, angstrom_to_bohr, bohr_to_angstrom, lam_to_omega, nm_to_au, fs_to_au, au_to_fs, au_to_Vpm, Vpm_to_au, Cm_to_au
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
        self.t_end = fs_to_au(t_end)
        self.time = np.linspace(0, t_end, n_steps, endpoint=False)
        self.n_steps = n_steps

    def define_bands(self, Ec, Ev, tc, tv):
        self.bands = BandStructure(Ec=Ec, Ev=Ev, tc=tc, tv=tv, a=bohr_to_angstrom(self.a))
    
    def define_pulse(self, sigma, lam, t_start, E0):
        """lam in nm, sigma in fs, Enull in V/m"""
        self.t_start = fs_to_au(t_start)
        self.sigma = fs_to_au(sigma)
        lam = nm_to_au(lam)
        self.omega = lam_to_omega(lam) 
        self.E0 = Vpm_to_au(E0)
        self.E_field= gaussian_sine(t=self.time, omega=self.omega, sigma=self.sigma, t_start=self.t_start, E0=self.E0)
        

    def define_system(self, num_k, a):
        """a in angstrom"""
        self.num_k = num_k
        self.a = angstrom_to_bohr(a)
        self.k_list = np.linspace(-np.pi/self.a, np.pi/self.a, num_k, endpoint=False)
        mat_init = np.zeros((len(self.k_list), 2, 2))
        mat_init[:,1,1] = 1 # fully populate conduction band
        self.mat_init = mat_init.flatten()

    def set_H_constant(self, dipole_element):
        dipole_element = Cm_to_au(dipole_element)
        h_const = np.zeros((len(self.k_list), 2, 2))
        for i,k in enumerate(self.k_list):
            h_const[i] = self.bands.get_H_mat(k)
        self.dipole_mat = np.zeros((2,2))
        self.dipole_mat[0,1] = dipole_element  # not part of H_null but constant over time
        self.dipole_mat[1,0] = dipole_element  # not part of H_null but constant over time
        h_const = h_const.flatten()
        self.h_const = h_const
    
    def get_H(self, t):
        h_const = np.reshape(self.h_const, (self.num_k, 2,2))
        E_mat = np.reshape(self.E_function(t=t), (self.num_k, 2,2))
        h_mat = E_mat * self.dipole_mat + h_const
        return h_mat.flatten()

    def commute(self, rho, t):
        H = np.reshape(self.get_H(t), (len(self.k_list), 2, 2)) 
        rho = np.reshape(rho, (len(self.k_list), 2, 2)) 
        commutator = np.einsum('ijk, ikl -> ijl', H,rho) - np.einsum('ijk, ikl -> ijl', rho, H)
        # commutator = H @ rho - rho @ H 
        return commutator.flatten()

    def get_rhs(self,t, rho):
        rho = self.y_to_rho(rho)
        rhs = -1j * self.commute(rho, t) + self.E_function(t) * self.get_k_partial(rho) #h_null is constant with time
        rhs = self.rho_to_y(rhs)
        return rhs 

    def E_function(self, t):
        """flat array of dim num_k to introduce the E-field for every k"""
        E = gaussian_sine(t, omega=self.omega, sigma=self.sigma, t_start=self.t_start, E0=self.E0)
        E = np.repeat(E, self.num_k)
        E_mat = np.zeros((self.num_k, 2,2))
        E_mat[:,1,0] = E
        E_mat[:,0,1] = E
        return E_mat.flatten()

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
        solution = solve_ivp(lambda rho, t: self.get_rhs(rho, t), t_span=(self.time[0], self.time[-1]), y0=self.rho_to_y(self.mat_init), t_eval=self.time,
                            #  method='Radau',
                            #   atol=1e-12, rtol=1e-12
                              )
        print(np.shape(solution.y))
        print(solution.status)
        rho_time = solution.y.T # transpose to switch time and other dimensions
        real = rho_time[:,:self.num_k * 4, ...]
        im = rho_time[:,self.num_k * 4 :, ...]
        rho_time = real + 1j*im
        self.solution = np.reshape(rho_time, (len(self.time),len(self.k_list), 2,2))

    def plot_field_E(self):
        fig, ax = plt.subplots()
        time = au_to_fs(self.time)
        E = au_to_Vpm(self.E_field)
        ax.plot(time, E)
        ax.set_xlabel("t/fs")
        ax.set_ylabel(r"E/$Vm^{-1}$")
        plt.show()
    
    def plot_density_matrix(self, k_index):
        fig, ax = plt.subplots()
        ax.plot(self.time, np.abs(self.solution[:,k_index,1,0])) 
        plt.show()

    def get_vector_potential(self):
        f = lambda t, y: gaussian_sine(t, self.omega, self.sigma, self.t_start, self.E0)
        solution = solve_ivp(f, (self.time[0], self.time[-1]), [0], t_eval=self.time,
                            #  method='DOP853', 
                            #  rtol=1e-10, 
                            #  atol=1e-12
                             )
        self.A_field = solution.y[0]

    def plot_field_A(self):
        fig, ax = plt.subplots()
        time = au_to_fs(self.time)
        A = au_to_fs(self.A_field)
        A = au_to_Vpm(A)
        ax.plot(time, self.A_field)
        ax.set_xlabel("t/fs")
        ax.set_ylabel(r"A/$Vsm^{-1}$")
        plt.show()
    
    def y_to_rho(self, y):
        real = y[:self.num_k*4]
        im = y[self.num_k*4:]
        rho = real + 1j * im
        return rho
    
    def rho_to_y(self, rho):
        real = np.real(rho)
        im = np.imag(rho)
        y = np.append(real, im, axis=0) 
        return y

    def get_heatmap_rho(self):
        rho = np.abs(self.solution)
        fig, axs = plt.subplots(2,1, figsize=(8,4))
        # time = np.linspace(0, self.t_end, self.n_steps+1, endpoint=False)
        # k_space = np.linspace(-np.pi/self.a, np.pi/self.a, self.num_k+1, endpoint=False)
        im1 = axs[0].pcolormesh(self.time, self.k_list, rho[:,:,0,0].T, shading='gouraud', label='valence band')
        im2 = axs[1].pcolormesh(self.time, self.k_list, rho[:,:,1,1].T, shading='gouraud', label='conduction band')
        axs[0].set_title('conduction band')
        axs[1].set_title('valence band')
        axs[0].set_xlabel('t')
        axs[0].set_ylabel('k')
        axs[1].set_xlabel('t')
        axs[1].set_ylabel('k')
        fig.colorbar(im1, ax=axs[0])
        fig.colorbar(im2, ax=axs[1])
        plt.legend()
        plt.show()



def gaussian_sine(t, omega, sigma, t_start, E0):
    return -E0 * np.sin(omega * t) * np.exp(-(t- t_start)**2 / (2 * sigma**2) )


if __name__ =="__main__":
    sim = Simulation(t_end=1000, n_steps=1000)
    sim.define_pulse(sigma=3, lam=774, t_start=11, E0=1e9) #E_0 = 1e11 roundabout corresponding to I = 1.5e14 W/cm^2
    sim.define_system(num_k=100, a=9.8) 
    sim.define_bands(Ec=4, Ev=-3, tc=-1.5, tv=0.5)
    sim.set_H_constant(dipole_element=9e-29) # corresponds to roundabout 9 a.u.
    print(Cm_to_au(9e-29))
    sim.integrate() 
    sim.plot_field_E()
    sim.get_heatmap_rho()

    
    

    

# independently: write circularly polarized light
# solve SBE ode integrate
# equidistant time steps
# nyquist theorem for time step estimation delta e delta t roundabout hbar
# write definiton vector potential based on E0, ignore envelope
# chekc for k dependence of simulation
# sum diagonal elements over k (integrate) plot over time, norm to electrons / volume cell
# calculate current

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
