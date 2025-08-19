import numpy as np
from scipy.interpolate import UnivariateSpline
from utilities import make_potential_unitcell, make_supercell, poeschl_teller, inner_prod
import os




class LCAOMatrices:

    def __init__(self, a, n_points, num_k, cached_int=False):
        self.a = a
        self.n_points = n_points
        self.k_list = np.linspace(-np.pi/a, np.pi/a, num_k)
        self.n_blocks = len(self.k_list)
        self.cached_int = cached_int

    def get_interals(self):
        self.integrals = LCAOAtomIntegrals(a=self.a, n_points=self.n_points, cached_int=self.cached_int)
        self.m_max = self.integrals.m_max
        self.integrals.create_potential()
        self.integrals.calc_S_mat()
        self.integrals.calc_H_mat()
        self.integrals.check_mat_symmetry(True)
        self.integrals.check_mat_symmetry(False)

    def _add_phase_integrals(self, k, m, n, mat):
        prefactors = [np.exp(1j * k * R) for R in range(-self.integrals.R_max, self.integrals.R_max + 1)]
        mat_elem = np.sum(prefactors* mat[:,m,n])
        return mat_elem 

    def _make_k_block(self, k, mat): # feed array atomic integrals and computes the k-block
        k_block = np.zeros((self.m_max, self.m_max), dtype='complex')
        for m in range(self.m_max):
            for n in range(self.m_max):
                k_block[m, n] = self._add_phase_integrals(k=k, m=m, n=n, mat=mat)
        return k_block

    def get_k_block_matrix(self): # get array of k_blocks of an operator in the u_LCAO basis 
        pass

    def get_Hk_orth(self):
        pass

    def get_Dk_orth(self):
        pass







class LCAOAtomIntegrals:

    def __init__(self, a, n_points, cached_int):
        self.R_max = 20
        self.m_max = 2
        self.a = a
        self.n_points = n_points # grid points per uni cell
        self.cached_int = cached_int
    
    def create_potential(self):
        x_u, V_u = make_potential_unitcell(lambda x : poeschl_teller(x, lam=5), n_points=self.n_points, a=self.a)
        self.x_space, self.V = make_supercell(x_u, V_u, n_super=self.R_max+8) # choose supercell big enough for all integrals until R_max
 
    def calc_S_mat(self):
        self.S_mat = np.zeros((self.R_max * 2 + 1, self.m_max, self.m_max)) 
        if self.cached_int and os.path.exists("S_mat.npy"):
            self.S_mat = np.load("S_mat.npy")
        else:
            self.S_mat = np.zeros((self.R_max * 2 + 1, self.m_max, self.m_max))
            for e, R in enumerate(range(-self.R_max, self.R_max +1)):
                for m in range(self.m_max):
                    for n in range(self.m_max):
                        self.S_mat[e, m, n] = self._two_center_int(m=m, n=n, R=R, operator=lambda x: x)
            np.save("S_mat.npy", self.S_mat)
                        
    def calc_R_mat(self):
        self.R_mat = np.zeros((self.R_max * 2 + 1, self.m_max, self.m_max)) 
        if self.cached_int and os.path.exists("R_mat.npy"):
            self.R_mat = np.load("R_mat.npy")
        else:
            self.R_mat = np.zeros((self.R_max * 2 + 1, self.m_max, self.m_max))
            for e, R in enumerate(range(-self.R_max, self.R_max +1)):
                for m in range(self.m_max):
                    for n in range(self.m_max):
                        self.R_mat[e, m, n] = self._two_center_int(m=m, n=n, R=R, operator=lambda x: x * self.x_space)
            np.save("R_mat.npy", self.R_mat)

    def calc_H_mat(self):
        self.H_mat = np.zeros((self.R_max * 2 + 1, self.m_max, self.m_max)) 
        if self.cached_int and os.path.exists("H_mat.npy"):
            self.R_mat = np.load("H_mat.npy")
        else:
            self.H_mat = np.zeros((self.R_max * 2 + 1, self.m_max, self.m_max))
            for e, R in enumerate(range(-self.R_max, self.R_max +1)):
                for m in range(self.m_max):
                    for n in range(self.m_max):
                        self.H_mat[e, m, n] = self._two_center_int(m=m, n=n, R=R, operator=lambda x: self._hamiltonian(x))
            np.save("H_mat.npy", self.H_mat)

    def _two_center_int(self, m, n, R, operator=lambda x: x): # does not converge to machine precision 0 for large distances
        x = self.x_space
        psi1 = shifted_function(R=0, m=m, a=self.a, x=x) # rewrite to avoid loading same funciton from file in every iteration of for loop
        psi2 = shifted_function(R=R, m=n, a=self.a, x=x) 
        return inner_prod(psi1, operator(psi2), x) 

    def _hamiltonian(self, psi):
        h = self.x_space[1] - self.x_space[0]
        laplacian = np.zeros(np.shape(psi))
        plus_h = np.roll(psi, 1)
        minus_h = np.roll(psi, -1)
        laplacian = (plus_h + minus_h - 2 * psi) / h**2 
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

    
def shifted_function(R, m, a, x):
    psi = np.load("numerov-five.npy")[m]
    xg = np.load("numerov-grid.npy")
    xg = xg + (R+4) * a
    spline = UnivariateSpline(xg, psi, s=0)
    psi_shift = spline(x)
    return psi_shift

if __name__ == "__main__":
    matrices = LCAOMatrices(a=2, n_points=100, num_k=50, cached_int=False)
    matrices.get_interals()