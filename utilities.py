import numpy as np
import matplotlib.pyplot as plt

def make_potential_unitcell(atom_pot, n_points, a, scale_H): # a is length of the unit cell, n_points the number of points that will be in each unit cell of the supercell
    n_wells = 100 
    x_space = np.linspace(0, n_wells * a, n_wells * n_points+1) # linspace guarantees even spacing for certain number of points
    V_tot = np.zeros(x_space.shape)
    for i in range(n_wells):
        V_tot += scale_H**2 * atom_pot(x_space - i * a)
    # plt.plot(x_space, V_tot)
    # plt.show()
    unit_start = int(n_wells * n_points * 0.5) # first ...
    unit_stop = int(n_points * (n_wells*0.5 + 1)) # ... and last element to cut out
    V_unit = V_tot[unit_start : unit_stop + 1] # second index is inclusive
    # V_unit += -np.max(V_unit)
    x_space = x_space[unit_start : unit_stop + 1] - x_space[unit_start]
    return x_space, V_unit # returns arrays where first and last elements correspond to lattice points

def make_supercell(x_space, V_unit, n_super):
    n_points = len(x_space) - 1
    max = x_space[-1]
    long_space = np.linspace(-max*n_super, max*n_super, n_points*n_super*2, endpoint=False)# dont repeat endpoint
    V_unit = V_unit[:-1]
    long_V = np.tile(V_unit, reps=n_super*2)
    # plt.plot(long_space, long_V)
    # plt.show()
    return long_space, long_V # returns array where the last point is not symmetry equivalent to first point

def poeschl_teller(xs, lam=5, a=1):
    return -lam * (lam + 1) * a**2 / (2 * np.cosh(a * xs) ** 2)

def inner_prod(psi1, psi2, x_space):
    h = x_space[1]- x_space[0]
    return np.trapezoid(psi1 * psi2, dx=h)


    
