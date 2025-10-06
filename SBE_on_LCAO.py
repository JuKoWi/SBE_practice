#! /usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.constants as constants
from unit_conversion import eV_to_au, au_to_ev, angstrom_to_bohr, bohr_to_angstrom, lam_to_omega, nm_to_au, fs_to_au, au_to_fs, au_to_Vpm, Vpm_to_au, Cm_to_au, au_to_A
from scipy.integrate import solve_ivp
from LCAO_for_SBE import LCAOAtomIntegrals, LCAOMatrices

class Simulation:
    def __init__(self, t_end, n_steps):
        """t-end in fs"""
        self.t_end = fs_to_au(t_end)
        self.time = np.linspace(0, self.t_end, n_steps, endpoint=False)
        self.n_steps = n_steps
        self.solution = None

    def define_pulse(self, sigma, lam, t_center, E0):
        """sigma and t_start in fs, E0 in V/m, lam in nm"""
        self.pulse = Field(time=self.time, sigma=sigma, lam=lam, t_center=t_center, E0=E0)

    def use_LCAO(self, num_k, a, scale_H, m_max, shift, scale2, vb_index, T2=0):
        self.T2 = fs_to_au(T2)
        self.a = angstrom_to_bohr(a)
        self.num_k = num_k
        
        matrices = LCAOMatrices(a=self.a, n_points=1000, num_k=num_k, m_max=m_max, scale_H=scale_H, shift=shift, scale2=scale2)
        self.m_basis = matrices.m_basis
        matrices.get_interals()
        matrices.get_H_blocks()
        matrices.get_S_blocks()
        matrices.get_nablak_blocks()
        matrices.get_transform_S()
        matrices.get_D_orth()
        matrices.get_H_orth()
        matrices.get_diagonalize_H()
        matrices.check_eigval()

        self.k_list = matrices.k_list
        
        self.X = matrices.diagonalize_H
        self.X_inv = matrices.diagonalize_H_dagger
        self.mat_init = np.zeros((self.num_k, self.m_basis, self.m_basis), dtype='complex')
        for i in range(vb_index):
            self.mat_init[:,i,i] = 1 # fully populate band
        self.mat_init = self.X @ self.mat_init @ self.X_inv
        self.h_const = matrices.H_orth
        self.dipole_mat = matrices.D_orth

    def get_H(self, t):
        h_mat = self.pulse.get_E(t) * self.dipole_mat + self.h_const
        return h_mat

    def commute(self, rho, t):
        H = self.get_H(t) 
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
        dephasing = 0
        if self.T2 != 0:
            transformed_rho = self.X_inv @ rho @ self.X
            dephasing = self.X @ ((1/self.T2) * (transformed_rho - transformed_rho * np.eye(self.m_basis))) @ self.X_inv
        k_deriv = self.get_k_partial(rho)
        rhs = 1j*self.commute(rho, t) + E * k_deriv  - dephasing 
        return self.rho_to_y(rhs) 

    def integrate(self):
        """needs self.rhs"""
        print('start integration')
        solution = solve_ivp(lambda t, y : self.get_rhs(t=t, y=y), t_span=(self.time[0], self.time[-1]), y0=self.rho_to_y(self.mat_init), t_eval=self.time,
                            # method='BDF', 
                            atol=1e-12, rtol=1e-12, # why does this resolve problems in boundary region?
                            )
        rho_time = solution.y.T # transpose to switch time and other dimensions
        self.solution = np.array([self.y_to_rho(rho_time[i]) for i in range(rho_time.shape[0])]) # rho(t) in orthogonal basis (not an energy eigenbasis)
    
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
        H_partial = self.get_k_partial(self.h_const)
        J = self.get_J(H_partial_k=H_partial)
        rho_J = self.get_rho_J(J=J, H_partial_k=H_partial) # rho * j(operator)
        self.current = np.einsum('ijkk -> i',rho_J) #trace over j rho
    
    def get_J(self, H_partial_k):
        """ j(t) = i [r, H(t)]"""
        dipole_mat = self.dipole_mat
        h_null = self.h_const
        J = - (H_partial_k - 1j * (np.einsum('kab, kbc -> kac', dipole_mat, h_null) - np.einsum('kab, kbc -> kac', h_null, dipole_mat)))
        return J

    def get_rho_J(self, J, H_partial_k):
        J = self.get_J(H_partial_k=H_partial_k)
        rho = self.solution
        return np.einsum('tkab,kbc -> tkac', rho, J)
    
    def calculate_spectrum(self):
        N = self.n_steps
        omega = np.array([np.exp(-1j * 2 * n *np.pi /N) for n in range(self.n_steps)])
        vandermonde = np.vander(omega, increasing=True).T
        result = vandermonde @ self.current
        ang_freq = 2 * np.pi * np.arange((self.n_steps)) / self.t_end
        energy_eV = au_to_ev(ang_freq)
        energy_eV = energy_eV[:len(energy_eV)//2]
        result = result[:len(result)//2]

        fix, ax = plt.subplots()
        def energy_to_harmonic(E):
            return E / au_to_ev(self.pulse.omega)
        
        def harmonic_to_energy(h):
            return h * au_to_ev(self.pulse.omega)

        ax.plot(energy_eV, np.log10(energy_eV**2 * np.abs(result)**2))
        ax.set_xlabel('E/eV')
        ax.set_ylabel(r'$\log_{10}(S)$')
        secax = ax.secondary_xaxis('top', functions=(energy_to_harmonic, harmonic_to_energy))
        secax.set_xlabel('Harmonic order')
        plt.show() #TODO auf 1 normieren
        

class Field:
    """must include a method with t as input"""

    def __init__(self, time, sigma, lam, t_center, E0):
        """lam in nm, sigma in fs, Enull in V/m"""
        self.time = time
        self.t_center = fs_to_au(t_center)
        self.sigma = fs_to_au(sigma)
        lam = nm_to_au(lam)
        self.omega = lam_to_omega(lam) 
        self.E0 = Vpm_to_au(E0)
        self.E_field = np.array([self.get_E(t) for t in self.time])

    def get_E(self, t):
        E = gaussian_sine(t, omega=self.omega, sigma=self.sigma, t_center=self.t_center, E0=self.E0)
        return E

    def get_vector_potential(self):
        f = lambda t, y: gaussian_sine(t, self.omega, self.sigma, self.t_center, self.E0)
        solution = solve_ivp(f, (self.time[0], self.time[-1]), [0], t_eval=self.time)
        self.A_field = solution.y[0]

class Plot:

    def __init__(self, simulation):
        self.m_basis = simulation.m_basis
        self.time = simulation.time
        self.solution = simulation.solution
        self.k_list = simulation.k_list
        self.E_field = simulation.pulse.E_field
        self.simulation = simulation
        self.X = simulation.X
        self.X_inv = simulation.X_inv
    
    def get_heatmap_rho(self):
        rho = np.abs(self.X_inv @ self.solution @ self.X)
        fig, axs = plt.subplots(self.m_basis+1,1, figsize=(8,4), sharex=True)
        for i in range(self.m_basis):
            im = axs[i].pcolormesh(au_to_fs(self.time), self.k_list, rho[:,:,i,i].T, shading='auto')
            fig.colorbar(im, ax=axs[i], orientation='horizontal')
        axs[self.m_basis].plot(au_to_fs(self.time), au_to_Vpm(self.E_field))
        axs[self.m_basis].set_xlabel('t / fs')
        axs[self.m_basis].set_ylabel(r'E / V $m^{-1}$')
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
        ax.plot(au_to_fs(self.time), np.sum(np.abs(self.solution[:,:,1,1]), axis=1)) 
        ax.set_ylabel(ylabel=r'n_{electrons, total}')
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

    def plot_bands(self):
        H = self.simulation.h_const
        k_list = self.simulation.k_list
        bands = np.zeros((self.simulation.num_k, self.m_basis))
        for i,k in enumerate(k_list):
            eigval, eigvec = la.eigh(H[i])
            bands[i] = eigval
        bands = bands.T
        plt.plot(k_list, bands[0])
        plt.plot(k_list, bands[1])
        plt.show()
    
    def plot_population(self):
        rho = self.X_inv @ self.solution @ self.X
        rho_no_k = np.sum(rho, axis=1)/np.shape(self.k_list)[0]
        for i in range(self.m_basis):
            plt.plot(self.time, rho_no_k[:,i,i])
        plt.show()

def gaussian_sine(t, omega, sigma, t_center, E0):
    return -E0 * np.sin(omega * t) * np.exp(-(t- t_center)**2 / (2 * sigma**2))


if __name__ =="__main__":
    sim = Simulation(t_end=80, n_steps=2000)
    sim.define_pulse(sigma=5, lam=740, t_center=40, E0=2e9) #E_0 = 1e11 roundabout corresponding to I = 1.5e14 W/cm^2
    sim.use_LCAO(num_k=1000, a=bohr_to_angstrom(20), scale_H=0.21, m_max=2, T2=5, shift=0.4, scale2=0.19, vb_index=2)
    sim.integrate() 
    results = Plot(sim)
    results.get_heatmap_rho()
    sim.get_current()
    results = Plot(sim)
    results.plot_current()
    sim.calculate_spectrum()
    results.plot_density_matrix(k_index=0)
