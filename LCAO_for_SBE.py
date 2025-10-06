import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.interpolate import UnivariateSpline
from utilities import make_potential_unitcell, make_supercell, poeschl_teller, inner_prod
import os
from scipy.linalg import eigh, ishermitian
from unit_conversion import Cm_to_au, eV_to_au
from numerov import solve_schroedinger, symmetric




class LCAOMatrices:
    """calculate integrals in orthogonal u basis for SBE"""

    def __init__(self, a, n_points, num_k, m_max, scale_H, scale2=1, shift=0):
        self.scale_H = scale_H
        if shift == 0:
            self.m_basis = m_max
        else: 
            self.m_basis = 2*m_max
        self.a = a
        self.n_points = n_points
        self.num_k = num_k
        self.k_list = np.linspace(-np.pi/a , np.pi/a, num_k, endpoint=False)
        self.n_blocks = len(self.k_list)
        self.shift = shift
        self.scale2 = scale2

    def get_interals(self):
        self.integrals = LCAOAtomIntegrals(a=self.a, n_points=self.n_points, m_basis=self.m_basis, scale_H=self.scale_H, scale2=self.scale2, shift=self.shift)
        self.integrals.get_atom_func()
        print('CALCULATE 2C INTEGRALS')
        self.integrals.create_potential()
        self.integrals.calc_S_mat()
        self.integrals.calc_H_mat()
        self.integrals.calc_R_mat()

    def _add_phase_integrals(self, k, m, n, mat):
        prefactors = np.array([np.exp(1j * k * R *self.a) for R in range(-self.integrals.R_max, self.integrals.R_max +1)], dtype='complex')
        mat_elem = np.sum(prefactors * mat[:,m,n]) 
        return mat_elem 

    def _make_k_block(self, k, mat): # feed array atomic integrals and computes the k-block
        k_block = np.zeros((self.m_basis, self.m_basis), dtype='complex')
        for m in range(self.m_basis):
            for n in range(self.m_basis):
                k_block[m, n] = self._add_phase_integrals(k=k, m=m, n=n, mat=mat)
        return k_block

    def get_S_blocks(self): 
        self.S_blocks = np.zeros((self.n_blocks, self.m_basis, self.m_basis), dtype='complex')
        for i,k in enumerate(self.k_list):
            S = self._make_k_block(k=k, mat=self.integrals.S_mat)
            self.S_blocks[i] = S

    def custom_S_blocks(self, k_list):
        """get S for another k-grid"""
        S_blocks = np.zeros((self.n_blocks, self.m_basis, self.m_basis), dtype='complex')
        for i,k in enumerate(k_list):
            S = self._make_k_block(k, mat=self.integrals.S_mat)
            S_blocks[i] = S
        return S_blocks

    def get_H_blocks(self):
        self.H_blocks = np.zeros((self.n_blocks, self.m_basis, self.m_basis), dtype='complex')
        for i,k in enumerate(self.k_list):
            H = self._make_k_block(k=k, mat=self.integrals.H_mat)
            self.H_blocks[i] = H

    def get_nablak_blocks(self):
        self.nablak_blocks = np.zeros((self.n_blocks, self.m_basis, self.m_basis), dtype='complex')
        for i,k in enumerate(self.k_list):
            R = self._make_k_block(k=k, mat=self.integrals.R_mat) # prefactor of 1/N?
            self.nablak_blocks[i] = R

    def get_transform_S(self):
        """get S^{-1/2}"""
        unitary = np.zeros((self.n_blocks, self.m_basis, self.m_basis), dtype='complex')
        diag = np.zeros((self.n_blocks, self.m_basis, self.m_basis), dtype='complex')
        U_dagger = np.zeros((self.n_blocks, self.m_basis, self.m_basis), dtype='complex')
        diag_minus_half = np.zeros_like(diag)
        for i,k in enumerate(self.k_list):
            eigval, eigvec = eigh(self.S_blocks[i]) # doublecheck, if transpose of U necessary
            unitary[i] = eigvec 
            diag[i] = np.diag(eigval)
            U_dagger[i] = eigvec.conj().T
            inv_sqrt_eigval = np.diag(1/np.sqrt(eigval))
            diag_minus_half[i] = inv_sqrt_eigval
        self.S_minus_half = np.einsum('kab, kbc, kcd -> kad', unitary, diag_minus_half, U_dagger) 

    def calc_k_partial(self, block_mat):
        xplush = np.roll(block_mat, shift=-1, axis=0)
        xminush = np.roll(block_mat, shift=1, axis=0)
        xplus2h= np.roll(block_mat, shift=-2, axis=0)
        xminus2h = np.roll(block_mat, shift=2, axis=0)
        h = self.k_list[1] - self.k_list[0]
        deriv = (8 * xplush - 8 * xminush + xminus2h - xplus2h)/(12 * h)
        return deriv
    
    def fine_S_partial(self):
        h = (self.k_list[1] - self.k_list[0]) /1000
        klist_plus = self.k_list + h
        klist_minus = self.k_list -h
        S_plus = self.custom_S_blocks(k_list=klist_plus)
        S_minus = self.custom_S_blocks(k_list=klist_minus)
        S_half_plus = self.get_transform_S_fine(k_list=klist_plus, S_blocks=S_plus) 
        S_half_minus = self.get_transform_S_fine(k_list=klist_minus, S_blocks=S_minus)
        klist_2plus = self.k_list + 2*h
        klist_2minus = self.k_list -2*h
        S_2plus = self.custom_S_blocks(k_list=klist_2plus)
        S_2minus = self.custom_S_blocks(k_list=klist_2minus)
        S_half_2plus = self.get_transform_S_fine(k_list=klist_2plus, S_blocks=S_2plus) 
        S_half_2minus = self.get_transform_S_fine(k_list=klist_2minus, S_blocks=S_2minus)
        deriv = (8* S_half_plus - 8 * S_half_minus + S_half_2minus - S_half_2plus)/(12 * h)
        return deriv

    def get_transform_S_fine(self, k_list, S_blocks):
        unitary = np.zeros((self.n_blocks, self.m_basis, self.m_basis), dtype='complex')
        diag = np.zeros((self.n_blocks, self.m_basis, self.m_basis), dtype='complex')
        U_dagger = np.zeros((self.n_blocks, self.m_basis, self.m_basis), dtype='complex')
        diag_minus_half = np.zeros_like(diag)
        for i,k in enumerate(k_list):
            eigval, eigvec = eigh(S_blocks[i]) # doublecheck, if transpose of U necessary
            unitary[i] = eigvec 
            diag[i] = np.diag(eigval)
            U_dagger[i] = eigvec.conj().T
            inv_sqrt_eigval = np.diag(1/np.sqrt(eigval))
            diag_minus_half[i] = inv_sqrt_eigval
        S_minus_half = np.einsum('kab, kbc, kcd -> kad', unitary, diag_minus_half, U_dagger) 
        return S_minus_half


    def get_D_orth(self):
        S_half = self.S_minus_half
        S_half_adj = S_half # is self adjoint 
        # self.D_orth = 1j*(S_half_adj @ self.S_blocks @ self.fine_S_partial() + S_half_adj @ self.nablak_blocks @ S_half ) #d_mn = i<u_mk|nabla k |u_nk>
        self.D_orth = 1j*(S_half_adj @ self.S_blocks @ self.calc_k_partial(S_half) + S_half_adj @ self.nablak_blocks @ S_half ) # k-derivative calculated on k_list-grid
        self.D_orth = 0.5* (np.transpose(self.D_orth, axes=(0,2,1)).conj() + self.D_orth) #make hermitian manually
        A = S_half_adj @ self.S_blocks @ self.calc_k_partial(S_half)
        B = S_half_adj @ self.nablak_blocks @ S_half 

    def get_H_orth(self):
        S_half = self.S_minus_half
        S_half_adj = S_half # is self adjoint 
        self.H_orth = S_half_adj @ self.H_blocks @ S_half 
        self.H_orth = 0.5* (np.transpose(self.H_orth, axes=(0,2,1)).conj() + self.H_orth)
        # for i,k in enumerate(self.k_list):
            # print(ishermitian(self.H_orth[i]))

    def get_diagonalize_H(self):
        unitary = np.zeros_like(self.H_orth)
        unitary_dagger = np.zeros_like(self.H_orth)
        eigval, unitary= eigh(self.H_orth)
        unitary_dagger = np.transpose(unitary, axes=(0,2,1)).conj()
        self.diagonalize_H = unitary
        self.diagonalize_H_dagger = unitary_dagger
    
    def matrix_plot(self, mat_blocks):
        fig, ax = plt.subplots(1,2)
        im1 = ax[0].imshow(np.abs(mat_blocks[0]), cmap='viridis')
        im2 = ax[1].imshow(np.angle(mat_blocks[0], deg=False), cmap='twilight')
        fig.colorbar(im1, ax=ax[0])
        fig.colorbar(im2, ax=ax[1])
        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03]) 
        k_slider = Slider(ax=slider_ax, label='k_value', valmin=0, valmax=len(self.k_list)-1, valinit=0, valstep=1)
        def update(val):
            idx = int(k_slider.val)
            im1.set_data(np.abs(mat_blocks[idx]))
            im2.set_data(np.angle(mat_blocks[idx], deg=False))
            ax[0].set_title(f"absolute for k = {self.k_list[idx]}")
            ax[1].set_title(f"phase for k = {self.k_list[idx]}")
            fig.canvas.draw_idle()
        k_slider.on_changed(update)
        plt.show()

    def check_eigval(self):
        bands = np.zeros((self.num_k,self.m_basis))
        for i,k in enumerate(self.k_list):
            # eig1, eigvec1 = eigh(self.H_blocks[i], self.S_blocks[i])
            eig2, eigvec2 = eigh(self.H_orth[i])
            bands[i] = eig2
            # print(np.isclose(eig1, eig2, rtol=1e-10))
        bands = bands.T
        for m in range(self.m_basis):
            plt.plot(self.k_list, bands[m])
        plt.show()
    
    def overwrite_matrices(self):
        self.D_orth = np.zeros_like(self.D_orth)
        self.H_orth = np.zeros_like(self.H_orth)
        Ec = eV_to_au(4)
        Ev = eV_to_au(-3)
        tc = eV_to_au(-1.5)
        tv = eV_to_au(0.5)
        for i,k in enumerate(self.k_list):
            self.D_orth[i,1,0] = Cm_to_au(9e-29)
            self.D_orth[i,0,1] = Cm_to_au(9e-29)
            self.H_orth[i,1,1] = Ec + tc * np.cos(k * self.a)
            self.H_orth[i,0,0] = Ev + tv * np.cos(k * self.a)
            # self.H_orth[i,2,2] = eV_to_au(10)
            # self.D_orth[i,0,2] = Cm_to_au(9e-29)
            # self.D_orth[i,2,1] = Cm_to_au(9e-29)
            # self.D_orth[i,2,0] = Cm_to_au(9e-29)
            # self.D_orth[i,1,2] = Cm_to_au(9e-29)
        unitary = np.zeros((self.num_k, self.m_basis, self.m_basis))
        for i, k in enumerate(self.k_list):
            theta = np.sin(self.a * k)
            unitary[i, 0,0] = np.cos(theta)
            unitary[i,0,1] = -np.sin(theta)
            unitary[i,1,0] = np.sin(theta)
            unitary[i,1,1] = np.cos(theta)
            # unitary[i,2,2] = 1
        # print(unitary)
        # print(np.transpose(unitary, axes=(0,2,1)))
        self.D_orth = np.transpose(unitary, axes=(0,2,1)) @ self.D_orth @ unitary
        self.H_orth = np.transpose(unitary, axes=(0,2,1)) @ self.H_orth @ unitary
    
    def plot_bands_directly(self):
        bands = np.zeros((self.num_k, self.m_basis))
        for i, k in enumerate(self.k_list):
            eigval, eigvec = eigh(self.H_blocks[i], self.S_blocks[i])
            bands[i] = eigval
        bands = bands.T
        for i in range(self.m_basis):
            plt.plot(self.k_list, bands[i])
        plt.show()

    def analyze_dipole(self, real=True):
        D = self.D_orth
        if real:
            D = np.real(self.D_orth)
        else:
            D = np.imag(self.D_orth)
        for m in range(self.m_basis):
            for n in range(self.m_basis):
                plt.plot(self.k_list, D[:,m,n], label=f"{m},{n}")
        plt.legend()
        plt.show()

    def shift_band(self):
        D = self.diagonalize_H_dagger @ self.H_orth @ self.diagonalize_H
        D[:,2,2] -= 1.2
        self.H_orth = self.diagonalize_H @ D @ self.diagonalize_H_dagger



class LCAOAtomIntegrals:
    """construct integrals between two atom orbitals"""

    def __init__(self, a, n_points, m_basis, scale_H, scale2, shift):
        self.scale_H = scale_H
        self.R_max = 4
        self.m_basis = m_basis 
        self.a = a 
        self.n_points = n_points # grid points per uni cell
        self.shift = shift
        self.scale2 = scale2

    def get_atom_func(self):
        N_single = 10000  # Number of grid points
        xmax_single = 70 # Extent of the grid
        lam = 5
        xg = np.linspace(0, xmax_single, N_single)
        h = xg[1] - xg[0]
        V1 = poeschl_teller(xg, lam=lam, a=self.scale_H)
        V2 = poeschl_teller(xg, lam=5, a=self.scale2)

        print('CALCULATE ATOM ORBITAL ENERGIES')
        def create_wf(k, gerade, V, Etry=-1):
            u, E, dE, n_nodes = solve_schroedinger(V, k=k, gerade=gerade, h=h, Etry=Etry)
            print(E)
            psi0 = symmetric(u, gerade=gerade)
            psi0 /= np.sqrt(np.trapezoid(psi0 * psi0, dx=h))
            return psi0

        k = 0
        ao = np.zeros((lam*2, N_single * 2 - 1)) 
        for i in range(lam): 
            gerade = (i%2 == 0)
            ao[i*2] = create_wf(k, gerade, V1)
            ao[i*2 +1] = create_wf(k, gerade, V2)
            if i%2 == 1:
                k +=1
        V_plot = symmetric(V1, gerade=True)
        xg = symmetric(xg, gerade=False)
        np.save("numerov-five.npy", arr=ao)
        np.save("numerov-grid.npy", xg)

        for i, psi in enumerate(ao):
            shifted_grid = xg + self.shift * self.a
            spline = UnivariateSpline(shifted_grid, psi, s=0)
            psi_shift = spline(xg)
            if (i%2) != 0:
                ao[i] = psi_shift
            

        self.grid = xg
        if self.shift == 0:
            self.atom_func = ao[::2]
        else: 
            self.atom_func = ao


        # fig, axs = plt.subplots(2, 1, sharex=False)

        # axs[0].plot(xg, V_plot)

        # for i,orb in enumerate(ao):
        #     axs[1].plot(xg, orb, label=f"psi{i} with E = {orb[1]:.2f}")

        # # axs[0].set_xlim((-5, 5))
        # # axs[1].set_xlim((-6, 6))

        # plt.legend(loc='upper right')
        # plt.show()

    def create_potential(self):
        x_u, V_u = make_potential_unitcell(n_points=self.n_points, a=self.a, scaleH=self.scale_H, scale2=self.scale2, shift=self.shift)
        self.x_space, self.V = make_supercell(x_u, V_u, n_super=self.R_max+2) # choose supercell big enough for all integrals until R_max
 
    def calc_S_mat(self):
        self.S_mat = np.zeros((self.R_max * 2 + 1, self.m_basis, self.m_basis)) 
        for i, R in enumerate(range(-self.R_max, self.R_max +1)):
            for m in range(self.m_basis):
                for n in range(self.m_basis):
                    self.S_mat[i, m, n] = self._two_center_int(m=m, n=n, R=R, operator=lambda x: x)
                        
    def calc_R_mat(self):
        self.R_mat = np.zeros((self.R_max * 2 + 1, self.m_basis, self.m_basis), dtype='complex') 
        for i, R in enumerate(range(-self.R_max, self.R_max +1)):
            for m in range(self.m_basis):
                for n in range(self.m_basis):
                    self.R_mat[i, m, n] = self._two_center_int(m=m, n=n, R=R, operator=lambda x: x * (self.x_space)) * 1j


    def calc_H_mat(self):
        self.H_mat = np.zeros((self.R_max * 2 + 1, self.m_basis, self.m_basis)) 
        for i, R in enumerate(range(-self.R_max, self.R_max +1)):
            for m in range(self.m_basis):
                for n in range(self.m_basis):
                    self.H_mat[i, m, n] = self._two_center_int(m=m, n=n, R=R, operator=lambda x: self._hamiltonian(x))
            if R == 0:
                print(self.H_mat[i])

    def _two_center_int(self, m, n, R, operator=lambda x: x): # does not converge to machine precision 0 for large distances
        x = self.x_space
        psi1 = self.shifted_function(R=R, m=m, a=self.a, x=x) # rewrite to avoid loading same funciton from file in every iteration of for loop
        psi2 = operator(self.shifted_function(R=0, m=n, a=self.a, x=x)) 
        # if R == 0:
        #     plt.plot(self.x_space, psi1)
        #     plt.plot(self.x_space, psi2)
        #     plt.show()
        return inner_prod(psi1, psi2, x) 

    def _hamiltonian(self, psi):
        h = self.x_space[1] - self.x_space[0]
        plus_h = np.roll(psi, 1) 
        minus_h = np.roll(psi, -1)
        plus_2h = np.roll(psi, 2)
        minus_2h = np.roll(psi, -2)
        laplacian = (-plus_2h +16*plus_h - 30 *psi + 16 * minus_h - minus_2h)/(12 * h**2) 
        H = - 0.5 * laplacian + self.V * psi
        return H
    
    def check_mat_symmetry(self, mat):
        diff = np.zeros((self.m_basis, self.m_basis))
        for m in range(self.m_basis):
            for n in range(self.m_basis):
                for R in range(0, self.R_max + 1):
                    diff[m,n] += np.abs(mat[R,m,n] - mat[-R-1,n,m])
        print(diff)

    def shifted_function(self, R, m, a, x):
        psi = self.atom_func[m]
        xg = self.grid
        xg = xg + R * a
        spline = UnivariateSpline(xg, psi, s=0)
        psi_shift = spline(x)
        return psi_shift

if __name__ == "__main__":
    matrices = LCAOMatrices(a=28, n_points=1000, num_k=1000, m_max=4, scale_H=0.21, shift=0.4, scale2=0.19) #good parameters: a=2.2, 2.3; 2.4; , scale_H=0.7; => 1.08 eV;1.5 eV; 4eV
    matrices.get_interals()
    matrices.get_H_blocks()
    matrices.get_S_blocks()
    matrices.get_nablak_blocks()
    matrices.get_transform_S()
    matrices.get_D_orth()
    matrices.get_H_orth()
    # matrices.matrix_plot(mat_blocks=matrices.D_orth)
    # matrices.matrix_plot(mat_blocks=matrices.H_orth)
    matrices.check_eigval()