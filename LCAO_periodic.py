import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from LCAO_on_numerov import inner_prod
from realspace_base_periodic import make_potential_unitcell, make_supercell
from LCAO_on_numerov import poeschl_teller
import os
from scipy.linalg import eigh, eig
from scipy.linalg import ishermitian




def shifted_function(R, m, a, x):
    psi = np.load("numerov-five.npy")[m]
    xg = np.load("numerov-grid.npy")
    xg = xg + (R+4) * a
    spline = UnivariateSpline(xg, psi, s=0)
    psi_shift = spline(x)
    return psi_shift

class LCAOIntegrals:
    def __init__(self, a, n_points, cached_int):
        self.R_max = 20
        self.m_max = 3
        self.a = a
        self.n_points = n_points #grid points per uni cell
        self.cached_int = cached_int

    def create_potential(self):
        x_u, V_u = make_potential_unitcell(lambda x : poeschl_teller(x, lam=5), n_points=self.n_points, a=self.a)
        self.x_space, self.V = make_supercell(x_u, V_u, n_super=self.R_max+8) # choose supercell big enough for all integrals until R_max
    
    def calc_S_mat(self):
        self.S_mat = np.zeros((self.R_max * 2 + 1, self.m_max, self.m_max)) 
        if self.cached_int and os.path.exists("S_mat.npy"):
            self.S_mat = np.load("S_mat.npy")
        else:
            # for e, R in enumerate(range(-self.R_max, self.R_max + 1)):
            #     for m in range(self.m_max):
            #         for n in range(self.m_max):
            #             self.S_mat[e, m, n] = self._two_center_int(m=m, n=n, R=R, hamilton=False)
            for R in range(0, self.R_max + 1):
                for m in range(self.m_max):
                    for n in range(self.m_max):
                        val = self._two_center_int(m=m, n=n, R=R, hamilton=False)
                        index1 = self.R_max - R
                        index2 = self.R_max + R
                        self.S_mat[index1, m, n] = val
                        self.S_mat[index2, n, m] = val


            np.save("S_mat.npy", self.S_mat)
    
    def calc_H_mat(self):
        if self.cached_int and os.path.exists("H_mat.npy"):
            self.H_mat = np.load("H_mat.npy")
        else:
            self.H_mat = np.zeros((self.R_max * 2 + 1, self.m_max, self.m_max))
            for e, R in enumerate(range(-self.R_max, self.R_max +1)):
                for m in range(self.m_max):
                    for n in range(self.m_max):
                        self.H_mat[e, m, n] = self._two_center_int(m=m, n=n, R=R, hamilton=True)
            # for e, R in range(0, self.R_max + 1):
            #     for m in range(self.m_max):
            #         for n in range(self.m_max):
            #             val = self._two_center_int(m=m, n=n, R=R, hamilton=False)
            #             index1 = self.R_max - R
            #             index2 = self.R_max + R
            #             self.H_mat[index1, m, n] = val
            #             self.H_mat[index2, n, m] = val
            np.save("H_mat.npy", self.H_mat)
        
    
    def _two_center_int(self, m, n, R, hamilton=False): # does not converge to machine precision 0 for large distances
        x = self.x_space
        psi1 = shifted_function(R=0, m=m, a=self.a, x=x) # rewrite to avoid loading same funciton from file in every iteration of for loop
        psi2 = shifted_function(R=R, m=n, a=self.a, x=x) # also use caching
        #plt.plot(self.x_space, psi1)
        # plt.plot(self.x_space, self.V)
        # #plt.plot(self.x_space, psi2)
        # plt.show()
        if hamilton:
            psi2 = self._hamiltonian(psi2)
        return inner_prod(psi1, psi2, x) # still non-machine precision value at high distances 

    def _hamiltonian(self, psi):
        h = self.x_space[1] - self.x_space[0]
        laplacian = np.zeros(np.shape(psi))
        psi_pad = np.pad(psi, pad_width=1, mode="constant", constant_values=0)
        laplacian = (psi_pad[:-2] + psi_pad[2:] - 2 * psi_pad[1:-1]) / h**2
        H = - 0.5 * laplacian + self.V * psi
        return H

    def check_mat_symmetry(self, hamilton):
        diff = np.zeros((self.m_max, self.m_max))
        if hamilton:
            for m in range(self.m_max):
                for n in range(self.m_max):
                    for R in range(0, self.R_max + 1):
                        index = R + self.R_max
                        diff[m,n] += np.abs(self.H_mat[R,m,n] - self.H_mat[-R-1,n,m])
        else:
            for m in range(self.m_max):
                for n in range(self.m_max):
                    for R in range(0, self.R_max + 1):
                        diff[m,n] += np.abs(self.S_mat[R,m,n] - self.S_mat[-R-1,n,m])
        print(diff)

class Crystal:

    def __init__(self, a, n_points, delta_k, cached_int=False):
        self.a = a
        self.n_points = n_points
        self.k_list = np.arange(-np.pi/a, np.pi/a, delta_k)
        self.n_blocks = len(self.k_list)
        self.cached_int = cached_int

    def get_interals(self):
        self.integrals = LCAOIntegrals(a=self.a, n_points=self.n_points, cached_int=self.cached_int)
        self.m_max = self.integrals.m_max
        self.integrals.create_potential()
        self.integrals.calc_S_mat()
        self.integrals.calc_H_mat()
        self.integrals.check_mat_symmetry(True)
        self.integrals.check_mat_symmetry(False)
        # print(self.integrals.S_mat)

    def solve_k_blocks(self):
        self.S_blocks = np.zeros((self.n_blocks, self.m_max, self.m_max), dtype='complex')
        self.H_blocks = np.zeros((self.n_blocks, self.m_max, self.m_max), dtype='complex')
        self.c_vecs = np.zeros((self.n_blocks, self.m_max, self.m_max), dtype='complex')
        self.eigvals = np.zeros((self.n_blocks, self.m_max))
        for i, k in enumerate(self.k_list):
            S = self._make_k_block(k=k, hamilton=False)
            self.S_blocks[i] = S
            print(ishermitian(S, atol=1e-12))
            H = self._make_k_block(k=k, hamilton=True)
            self.H_blocks[i] = H 
            # print(ishermitian(H, atol=1e-7))
            # print(np.real(S))
            # print(np.imag(S))
            E, vec = eigh(H,S)
            """ for small unit cells the The leading minor of order 5 of B is not positive definite. 
            The factorization of B could not be completed and no eigenvalues or eigenvectors were computed."""
            self.eigvals[i] = E
            self.c_vecs[i] = vec.T
    
    def plot_bands(self):
        fig, ax = plt.subplots(1,1)
        for m in range(self.m_max):
            ax.plot(self.k_list, self.eigvals[:,m], marker='o', markersize=1, linestyle='', label=f'band index: {m}')
        ax.set_xlabel('k')
        ax.set_ylabel('E/E_h')
        ax.legend()
        plt.show()
    
    def _add_phase_integrals(self, k, m, n, hamilton=False):
        if hamilton:
            prefactors = [np.exp(1j * k * R) for R in range(-self.integrals.R_max, self.integrals.R_max + 1)]
            mat_elem = np.dot(prefactors, self.integrals.H_mat[:,m,n])
        else:
            prefactors = [np.exp(1j * k * R) for R in range(-self.integrals.R_max, self.integrals.R_max + 1)]
            mat_elem = np.sum(prefactors* self.integrals.S_mat[:,m,n])
        return mat_elem 


    def _make_k_block(self, k, hamilton=False):
        mat = np.zeros((self.m_max, self.m_max), dtype='complex')
        for m in range(self.m_max):
            for n in range(self.m_max):
                mat[m, n] = self._add_phase_integrals(k=k, m=m, n=n, hamilton=hamilton)
        return mat

# use einsum
# use ft for phase factors



    

if __name__ == "__main__":
    crystal = Crystal(a=2, n_points=100, delta_k=0.01, cached_int=False)
    crystal.get_interals()
    crystal.solve_k_blocks()
    crystal.plot_bands()
