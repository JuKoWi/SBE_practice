import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from utilities import make_potential_unitcell, make_supercell, poeschl_teller, inner_prod
import os
from scipy.linalg import eigh, ishermitian
from unit_conversion import Cm_to_au, eV_to_au




class LCAOMatrices:
    """calculate integrals in orthogonal u basis for SBE"""

    def __init__(self, a, n_points, num_k, m_max, scale_H, cached_int=False):
        self.scale_H = scale_H
        self.m_max = m_max
        self.a = a
        self.n_points = n_points
        self.num_k = num_k
        self.k_list = np.linspace(-np.pi/a , np.pi/a, num_k, endpoint=False)
        self.n_blocks = len(self.k_list)
        self.cached_int = cached_int

    def get_interals(self):
        self.integrals = LCAOAtomIntegrals(a=self.a, n_points=self.n_points, m_max=self.m_max, scale_H=self.scale_H, cached_int=self.cached_int)
        self.integrals.create_potential()
        self.integrals.calc_S_mat()
        self.integrals.calc_H_mat()
        self.integrals.calc_R_mat()

    def _add_phase_integrals(self, k, m, n, mat):
        prefactors = np.array([np.exp(1j * k * R *self.a) for R in range(-self.integrals.R_max, self.integrals.R_max +1)], dtype='complex')
        mat_elem = np.sum(prefactors * mat[:,m,n]) 
        return mat_elem 

    def _make_k_block(self, k, mat): # feed array atomic integrals and computes the k-block
        k_block = np.zeros((self.m_max, self.m_max), dtype='complex')
        for m in range(self.m_max):
            for n in range(self.m_max):
                k_block[m, n] = self._add_phase_integrals(k=k, m=m, n=n, mat=mat)
        return k_block

    def get_S_blocks(self): 
        self.S_blocks = np.zeros((self.n_blocks, self.m_max, self.m_max), dtype='complex')
        for i,k in enumerate(self.k_list):
            S = self._make_k_block(k=k, mat=self.integrals.S_mat)
            self.S_blocks[i] = S
            # print(ishermitian(S, atol=1e-14))

    def get_H_blocks(self):
        self.H_blocks = np.zeros((self.n_blocks, self.m_max, self.m_max), dtype='complex')
        for i,k in enumerate(self.k_list):
            H = self._make_k_block(k=k, mat=self.integrals.H_mat)
            self.H_blocks[i] = H
            # print(ishermitian(H, atol=1e-13))

    def get_nablak_blocks(self):
        self.nablak_blocks = np.zeros((self.n_blocks, self.m_max, self.m_max), dtype='complex')
        for i,k in enumerate(self.k_list):
            R = self._make_k_block(k=k, mat=self.integrals.R_mat) # prefactor of 1/N?
            self.nablak_blocks[i] = R
            # print(R)
            # print(ishermitian(R, atol=1e-13))

    def get_transform_S(self):
        """get S^{-1/2}"""
        unitary = np.zeros((self.n_blocks, self.m_max, self.m_max), dtype='complex')
        diag = np.zeros((self.n_blocks, self.m_max, self.m_max), dtype='complex')
        U_dagger = np.zeros((self.n_blocks, self.m_max, self.m_max), dtype='complex')
        diag_minus_half = np.zeros_like(diag)
        for i,k in enumerate(self.k_list):
            eigval, eigvec = eigh(self.S_blocks[i]) # doublecheck, if transpose of U necessary
            unitary[i] = eigvec 
            diag[i] = np.diag(eigval)
            U_dagger[i] = eigvec.conj().T
            inv_sqrt_eigval = np.diag(1/np.sqrt(eigval))
            diag_minus_half[i] = inv_sqrt_eigval
            # print(np.allclose(np.transpose(unitary[i].conj(), axes=(1,0))@ unitary[i], np.eye(self.m_max), atol=1e-12))
        # test = np.einsum('kab, kbc -> kac', unitary, U_dagger)
        # print(test)
        # UDUdagger = np.einsum('kab, kbc, kcd -> kad', unitary, diag, U_dagger)
        # print(np.allclose(UDUdagger, self.S_blocks))
        # plt.plot(self.k_list, diag[:,1,1])
        # plt.plot(self.k_list, diag[:,0,0])
        # plt.show()
        self.S_minus_half = np.einsum('kab, kbc, kcd -> kad', unitary, diag_minus_half, U_dagger) 
        # print(self.S_minus_half[30])
        # print(f"S-12 * S-12 * S = 1 {np.allclose(np.eye(self.m_max),np.einsum('kab, kbc, kcd -> kad', self.S_minus_half, self.S_minus_half, self.S_blocks))}") 
    

    def calc_k_partial(self, block_mat):
        xplush = np.roll(block_mat, shift=-1, axis=0)
        xminush = np.roll(block_mat, shift=1, axis=0)
        xplus2h= np.roll(block_mat, shift=-2, axis=0)
        xminus2h = np.roll(block_mat, shift=2, axis=0)
        h = self.k_list[1] - self.k_list[0]
        deriv = (8 * xplush - 8 * xminush + xminus2h - xplus2h)/(12 * h)
        return deriv
    
    def get_D_orth(self):
        S_half = self.S_minus_half
        S_half_adj = S_half # is self adjoint 
        # for i,k in enumerate(self.k_list):
        #     print(ishermitian(S_half[i]))
        #     print(S_half[i] @ S_half_adj[i])
        self.D_orth = 1j*(S_half_adj @ self.S_blocks @ self.calc_k_partial(S_half) + S_half_adj @ self.nablak_blocks @ S_half ) #d_mn = i<u_mk|nabla k |u_nk>
        for m in range(self.m_max):
            for n in range(m):
                self.D_orth[:,n,m] = np.real(self.D_orth[:,m,n]) - 1j * np.imag(self.D_orth[:,m,n])
            self.D_orth[:,m,m] = np.real(self.D_orth[:,m,m])
        # for i,k in enumerate(self.k_list):
        #     print(ishermitian(self.D_orth[i]))

    def get_H_orth(self):
        S_half = self.S_minus_half
        S_half_adj = S_half # is self adjoint 
        # print(self.H_blocks)
        self.H_orth = S_half_adj @ self.H_blocks @ S_half 

    def get_diagonalize_H(self):
        unitary = np.zeros_like(self.H_orth)
        unitary_dagger = np.zeros_like(self.H_orth)
        for i, k in enumerate(self.k_list):
            eigval, eigvec = eigh(self.H_orth[i]) # doublecheck, if transpose of U necessary
            unitary[i] = eigvec
            unitary_dagger[i] = eigvec.conj().T
        self.diagonalize_H = unitary
        self.diagonalize_H_dagger = unitary_dagger


    def check_eigval(self):
        bands = np.zeros((self.num_k,self.m_max))
        for i,k in enumerate(self.k_list):
            eig1, eigvec1 = eigh(self.H_blocks[i], self.S_blocks[i])
            eig2, eigvec2 = eigh(self.H_orth[i])
            bands[i] = eig2
            # print(np.isclose(eig1, eig2, rtol=1e-10))
        bands = bands.T
        for m in range(self.m_max):
            plt.plot(self.k_list, bands[m])
        plt.show()
    
    def overwrite_matrices(self):
        self.D_orth = np.zeros_like(self.D_orth)
        # self.H_orth = np.zeros_like(self.H_orth)
        Ec = eV_to_au(4)
        Ev = eV_to_au(-3)
        tc = eV_to_au(-1.5)
        tv = eV_to_au(0.5)
        for i,k in enumerate(self.k_list):
            self.D_orth[i,1,0] = Cm_to_au(9e-29)
            self.D_orth[i,0,1] = Cm_to_au(9e-29)
            # self.H_orth[i,1,1] = Ec + tc * np.cos(k * self.a)
            # self.H_orth[i,0,0] = Ev + tv * np.cos(k * self.a)
            # self.H_orth[i,2,2] = eV_to_au(10)
            # self.D_orth[i,0,2] = Cm_to_au(9e-29)
            self.D_orth[i,2,1] = Cm_to_au(9e-29)
            # self.D_orth[i,3,0] = Cm_to_au(9e-29)
            self.D_orth[i,1,2] = Cm_to_au(9e-29)
        # unitary = np.zeros((self.num_k, self.m_max, self.m_max))
        # for i, k in enumerate(self.k_list):
        #     par_k = np.sin(self.a * k)
        #     unitary[i, 0,0] = np.cos(par_k)
        #     unitary[i,0,1] = -np.sin(par_k)
        #     unitary[i,1,0] = np.sin(par_k)
        #     unitary[i,1,1] = np.cos(par_k)
        #     unitary[i,2,2] = 1
        # # print(unitary)
        # # print(np.transpose(unitary, axes=(0,2,1)))
        # self.D_orth = np.transpose(unitary, axes=(0,2,1)) @ self.D_orth @ unitary
        # self.H_orth = np.transpose(unitary, axes=(0,2,1)) @ self.H_orth @ unitary
    
    def plot_bands_directly(self):
        bands = np.zeros((self.num_k, self.m_max))
        for i, k in enumerate(self.k_list):
            eigval, eigvec = eigh(self.H_blocks[i], self.S_blocks[i])
            bands[i] = eigval
        bands = bands.T
        for i in range(self.m_max):
            plt.plot(self.k_list, bands[i])
        plt.show()

    def analyze_dipole(self, real=True):
        D = self.D_orth
        if real:
            D = np.real(self.D_orth)
        else:
            D = np.imag(self.D_orth)
        for m in range(self.m_max):
            for n in range(self.m_max):
                plt.plot(self.k_list, D[:,m,n], label=f"{m},{n}")
        plt.legend()
        plt.show()

    def shift_band(self):
        D = self.diagonalize_H_dagger @ self.H_orth @ self.diagonalize_H
        D[:,2,2] -= 1.2
        self.H_orth = self.diagonalize_H @ D @ self.diagonalize_H_dagger



class LCAOAtomIntegrals:
    """construct integrals between two atom orbitals"""

    def __init__(self, a, n_points, m_max, cached_int, scale_H):
        self.scale_H = scale_H
        self.R_max = 4
        self.m_max = m_max 
        self.a = a 
        self.n_points = n_points # grid points per uni cell
        self.cached_int = cached_int
    
    def create_potential(self):
        x_u, V_u = make_potential_unitcell(
            # lambda x:-5/((x*10)**2+ 1),
            lambda x : poeschl_teller(x, lam=5), 
            n_points=self.n_points, a=self.a, scale_H=self.scale_H
        )
        self.x_space, self.V = make_supercell(x_u, V_u, n_super=self.R_max+2) # choose supercell big enough for all integrals until R_max
 
    def calc_S_mat(self):
        self.S_mat = np.zeros((self.R_max * 2 + 1, self.m_max, self.m_max)) 
        if self.cached_int and os.path.exists("S_mat.npy"):
            self.S_mat = np.load("S_mat.npy")
        else:
            for i, R in enumerate(range(-self.R_max, self.R_max +1)):
                for m in range(self.m_max):
                    for n in range(self.m_max):
                        self.S_mat[i, m, n] = self._two_center_int(m=m, n=n, R=R, operator=lambda x: x)
            np.save("S_mat.npy", self.S_mat)
                        
    def calc_R_mat(self):
        self.R_mat = np.zeros((self.R_max * 2 + 1, self.m_max, self.m_max), dtype='complex') 
        if self.cached_int and os.path.exists("R_mat.npy"):
            self.R_mat = np.load("R_mat.npy")
        else:
            for i, R in enumerate(range(-self.R_max, self.R_max +1)):
                for m in range(self.m_max):
                    for n in range(self.m_max):
                        self.R_mat[i, m, n] = self._two_center_int(m=m, n=n, R=R, operator=lambda x: x * (self.x_space)) * 1j

            np.save("R_mat.npy", self.R_mat)

    def calc_H_mat(self):
        self.H_mat = np.zeros((self.R_max * 2 + 1, self.m_max, self.m_max)) 
        if self.cached_int and os.path.exists("H_mat.npy"):
            self.H_mat = np.load("H_mat.npy")
        else:
            self.H_mat = np.zeros((self.R_max * 2 + 1, self.m_max, self.m_max))
            for i, R in enumerate(range(-self.R_max, self.R_max +1)):
                for m in range(self.m_max):
                    for n in range(self.m_max):
                        self.H_mat[i, m, n] = self._two_center_int(m=m, n=n, R=R, operator=lambda x: self._hamiltonian(x))
            np.save("H_mat.npy", self.H_mat)

    def _two_center_int(self, m, n, R, operator=lambda x: x): # does not converge to machine precision 0 for large distances
        x = self.x_space
        psi1 = shifted_function(R=R, m=m, a=self.a, x=x) # rewrite to avoid loading same funciton from file in every iteration of for loop
        psi2 = operator(shifted_function(R=0, m=n, a=self.a, x=x)) 
        # plt.plot(self.x_space, psi1)
        # plt.plot(self.x_space, psi2)
        # plt.show()
        return inner_prod(psi1, psi2, x) 

    def _hamiltonian(self, psi):
        h = self.x_space[1] - self.x_space[0]
        laplacian = np.zeros(np.shape(psi))
        plus_h = np.roll(psi, 1) 
        minus_h = np.roll(psi, -1)
        laplacian = (plus_h + minus_h - 2 * psi) / h**2 
        H = - 0.5 * laplacian + self.V * psi
        return H
    
    def check_mat_symmetry(self, mat):
        diff = np.zeros((self.m_max, self.m_max))
        for m in range(self.m_max):
            for n in range(self.m_max):
                for R in range(0, self.R_max + 1):
                    diff[m,n] += np.abs(mat[R,m,n] - mat[-R-1,n,m])
        print(diff)

    
def shifted_function(R, m, a, x):
    psi = np.load("numerov-five.npy")[m]
    xg = np.load("numerov-grid.npy")
    xg = xg + R * a
    spline = UnivariateSpline(xg, psi, s=0)
    psi_shift = spline(x)
    return psi_shift

if __name__ == "__main__":
    matrices = LCAOMatrices(a=2.5, n_points=200, num_k=100, m_max=4, scale_H=1, cached_int=False) #good parameters: a=2.5 for lam=5
    matrices.get_interals()
    matrices.get_H_blocks()
    matrices.get_S_blocks()
    # matrices.get_nablak_blocks()
    # matrices.get_transform_S()
    # matrices.get_D_orth()
    # matrices.get_H_orth()
    # # matrices.check_eigval()
    matrices.plot_bands_directly()