#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.constants as constants
from unit_conversion import eV_to_au, angstrom_to_bohr, bohr_to_angstrom, lam_to_omega, nm_to_au, fs_to_au, au_to_fs, au_to_Vpm, Vpm_to_au, Cm_to_au, au_to_A
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
    
    def get_H_partial_k(self, k):
        H = np.zeros((2,2))
        H[0,0] = -self.tc * self.a * np.sin(k * self.a)
        H[1,1] = -self.tv * self.a * np.sin(k * self.a)
        return H

class Simulation:
    """rewrite such that pulse and bands are provided as the respective classes
    only time interval /discretization for initialization"""
    def __init__(self, t_end, n_steps):
        """t-end in fs"""
        self.t_end = fs_to_au(t_end)
        self.time = np.linspace(0, self.t_end, n_steps, endpoint=False)
        self.n_steps = n_steps
        self.solution = None

    def define_bands(self, Ec, Ev, tc, tv):
        self.bands = BandStructure(Ec=Ec, Ev=Ev, tc=tc, tv=tv, a=bohr_to_angstrom(self.a))

    def define_pulse(self, sigma, lam, t_start, E0):
        """sigma and t_start in fs, E0 in V/m, lam in nm"""
        self.pulse = Field(time=self.time, sigma=sigma, lam=lam, t_start=t_start, E0=E0)
    
    def define_system(self, num_k, a):
        """a in angstrom"""
        self.num_k = num_k
        self.a = angstrom_to_bohr(a)
        self.k_list = np.linspace(-np.pi/self.a, np.pi/self.a, num_k, endpoint=False)
        self.mat_init = np.zeros((self.num_k, 2, 2), dtype='complex')
        self.mat_init[:,1,1] = 1 # fully populate valence band

    def set_H_constant(self, dipole_element):
        dipole_element = Cm_to_au(dipole_element)
        self.h_const = np.zeros((self.num_k, 2, 2))
        for i,k in enumerate(self.k_list):
            self.h_const[i] = self.bands.get_H_mat(k)
        self.dipole_mat = np.zeros((2,2))
        self.dipole_mat[0,1] = dipole_element  # not part of H_null but constant over time
        self.dipole_mat[1,0] = dipole_element  # not part of H_null but constant over time
    
    def get_H(self, t):
        # h_mat = np.repeat(np.expand_dims(self.get_E(t) * self.dipole_mat, axis=0), self.num_k, axis=0) + self.h_const
        h_mat = self.pulse.get_E(t) * self.dipole_mat + self.h_const
        return h_mat

    def commute(self, rho, t):
        H = self.get_H(t) 
        # commutator = np.einsum('ijk, ikl -> ijl', H,rho) - np.einsum('ijk, ikl -> ijl', rho, H)
        commutator = H @ rho - rho @ H 
        return commutator

    def get_k_partial(self, rho):
        xplush = np.roll(rho, shift=-1, axis=0)
        xminush = np.roll(rho, shift=1, axis=0)
        xplustwoh = np.roll(rho, shift=-2, axis=0)
        xminustwoh = np.roll(rho, shift=2, axis=0)
        h = self.k_list[1] - self.k_list[0]
        deriv = (8 * xplush - 8 * xminush + xminustwoh - xplustwoh)/(12 * h)
        return deriv

    def get_rhs(self,t, y):
        rho = self.y_to_rho(y)
        E = self.pulse.get_E(t) 
        rhs = 1j*self.commute(rho, t) + E * self.get_k_partial(rho) 
        return self.rho_to_y(rhs) 

    def integrate(self):
        """needs self.rhs"""
        solution = solve_ivp(lambda t, y : self.get_rhs(t=t, y=y), t_span=(self.time[0], self.time[-1]), y0=self.rho_to_y(self.mat_init), t_eval=self.time,
                            # method='BDF', 
                            # atol=1e-12, rtol=1e-12,
                            )
        rho_time = solution.y.T # transpose to switch time and other dimensions
        self.solution = np.array([self.y_to_rho(rho_time[i]) for i in range(rho_time.shape[0])])
    
    def y_to_rho(self, y):
        """takes matrix shape and returns flat shape with double the length"""
        return np.reshape(y[::2] + 1j * y[1::2], np.shape(self.mat_init))
    
    def rho_to_y(self, rho):
        y = np.zeros(rho.size *2)
        y[::2] = np.real(rho).flatten()
        y[1::2] = np.imag(rho).flatten()
        return y
    
    def check_idempotence(self):
        test_array = np.empty(np.shape(self.mat_init), dtype='complex')
        result = self.y_to_rho(self.rho_to_y(test_array))
        print(np.array_equal(test_array, result))

    def get_current(self):
        H_partial = self.get_H_partial()
        J = self.get_J(H_partial_k=H_partial)
        rho_J = self.get_rho_J(J=J)
        self.current = np.einsum('ijkk -> i',rho_J)
    
    def get_H_partial(self):
        H_partial = np.zeros((self.num_k, 2, 2))
        for i,k in enumerate(self.k_list):
            H_partial[i] = self.bands.get_H_partial_k(k)
        return H_partial

    def get_J(self, H_partial_k):
        dipole_mat = self.dipole_mat
        h_null = self.h_const
        J = - (H_partial_k - 1j * (np.einsum('ab,kbc -> kac', dipole_mat, h_null) - np.einsum('kab, bc -> kac', h_null, dipole_mat)))
        return J

    def get_rho_J(self, J):
        J = self.get_J(H_partial_k=self.get_H_partial())
        rho = self.solution
        return np.einsum('tkab,kbc -> tkac', rho, J)
    
    def calculate_spectrum(self):
        N = self.n_steps
        omega = np.array([np.exp(-1j * 2 * n *np.pi /N) for n in range(self.n_steps)])
        vandermonde = np.vander(omega, increasing=True).T
        result = vandermonde @ self.current
        ang_freq = 2 * np.pi * np.arange((self.n_steps)) / self.t_end
        plt.plot(ang_freq, np.log(result))
        plt.show()
        # write in ev, number harmonic
        




class Field:
    """must include a method with t as input"""

    def __init__(self, time, sigma, lam, t_start, E0):
        """lam in nm, sigma in fs, Enull in V/m"""
        self.time = time
        self.t_start = fs_to_au(t_start)
        self.sigma = fs_to_au(sigma)
        lam = nm_to_au(lam)
        self.omega = lam_to_omega(lam) 
        self.E0 = Vpm_to_au(E0)
        self.E_field = np.array([self.get_E(t) for t in self.time])

    def get_E(self, t):
        E = gaussian_sine(t, omega=self.omega, sigma=self.sigma, t_start=self.t_start, E0=self.E0)
        return E

    def get_vector_potential(self):
        f = lambda t, y: gaussian_sine(t, self.omega, self.sigma, self.t_start, self.E0)
        solution = solve_ivp(f, (self.time[0], self.time[-1]), [0], t_eval=self.time)
        self.A_field = solution.y[0]

class Plot:

    def __init__(self, simulation):
        self.time = simulation.time
        self.solution = simulation.solution
        self.k_list = simulation.k_list
        self.E_field = simulation.pulse.E_field
        self.simulation = simulation
    
    def get_heatmap_rho(self):
        rho = np.abs(self.solution)
        fig, axs = plt.subplots(3,1, figsize=(8,4), sharex=True)
        im1 = axs[0].pcolormesh(au_to_fs(self.time), self.k_list, rho[:,:,0,1].T, shading='auto')
        im2 = axs[1].pcolormesh(au_to_fs(self.time), self.k_list, rho[:,:,1,1].T, shading='auto')
        axs[2].plot(au_to_fs(self.time), au_to_Vpm(self.E_field))
        axs[0].set_title('conduction band')
        axs[1].set_title('valence band')
        axs[0].set_xlabel('t')
        axs[0].set_ylabel('k')
        axs[1].set_xlabel('t')
        axs[1].set_ylabel('k')
        fig.colorbar(im1, ax=axs[0], orientation='horizontal')
        fig.colorbar(im2, ax=axs[1], orientation='horizontal')
        plt.show()
    
    def plot_field_E(self):
        fig, ax = plt.subplots()
        time = au_to_fs(self.time)
        ax.plot(time, au_to_Vpm(self.E_field))
        ax.set_xlabel("t/fs")
        ax.set_ylabel(r"E/$Vm^{-1}$")
        plt.show()

    def plot_density_matrix(self, k_index):
        fig, ax = plt.subplots()
        ax.plot(au_to_fs(self.time), np.abs(self.solution[:,k_index,1,1])) 
        plt.show()

    def plot_field_A(self):
        A = au_to_fs(au_to_Vpm(self.simulation.pulse.A_field))
        fig, ax = plt.subplots()
        time = au_to_fs(self.time)
        ax.plot(time, A)
        ax.set_xlabel("t/fs")
        ax.set_ylabel(r"A/$Vsm^{-1}$")
        plt.show()
    
    def plot_current(self):
        J = au_to_A(np.real(self.simulation.current))
        time = au_to_fs(self.time)
        fig, ax = plt.subplots()
        ax.plot(time, J)
        ax.set_xlabel("t / fs")
        ax.set_ylabel("j / A")
        plt.show()

    


def gaussian_sine(t, omega, sigma, t_start, E0):
    return -E0 * np.sin(omega * t) * np.exp(-(t- t_start)**2 / (2 * sigma**2) )


if __name__ =="__main__":
    sim = Simulation(t_end=90, n_steps=5000)
    sim.define_pulse(sigma=5, lam=774, t_start=50, E0=1e9) #E_0 = 1e11 roundabout corresponding to I = 1.5e14 W/cm^2
    sim.define_system(num_k=100, a=9.8) 
    sim.define_bands(Ec=4, Ev=-3, tc=-1.5, tv=0.5)
    sim.set_H_constant(dipole_element=9e-29) # 9e-29 corresponds to roundabout 9 a.u.
    sim.integrate() 
    sim.get_current()
    results = Plot(sim)
    # results.plot_field_E()
    # sim.pulse.get_vector_potential()
    # results.plot_field_A()
    results.get_heatmap_rho()
    results.plot_current()
    sim.calculate_spectrum()
    # sim.plot_density_matrix(k_index=0)



    
    

    
"""
independently: write circularly polarized light
solve SBE ode integrate
equidistant time steps
nyquist theorem for time step estimation delta e delta t roundabout hbar
write definiton vector potential based on E0, ignore envelope
chekc for k dependence of simulation
sum diagonal elements over k (integrate) plot over time, norm to electrons / volume cell
dephasing
distort symmetry with phase on dipole
"""
