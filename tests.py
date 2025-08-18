from SBE import BandStructure, Simulation, Plot
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import ishermitian


"""
possible tests:
    check derivative for initial rho(0) with respect to k
    plot H_const over k
    plot H_const derivative over k
    plot E field
    check commutator 0 for rho(0) and no field
    leave out field: only constant matrix elements after integration
    trace constant rho_k
    0 leq diagonal elem leq 1
    check rho hermitian
"""

def check_rho_zero_derivative():
    sim = Simulation(t_end=30, n_steps=1000)
    sim.define_pulse(sigma=3, lam=774, t_start=11, E0=1)
    sim.define_system(num_k=100, a=9.8) 
    sim.define_bands(Ec=4, Ev=-3, tc=-1.5, tv=0.5)
    mat_init = sim.mat_init
    plt.plot(sim.k_list, mat_init[:,0,0], label='0,0')
    plt.plot(sim.k_list, mat_init[:,1,0], label='1,0')
    plt.plot(sim.k_list, mat_init[:,0,1], label='0,1')
    plt.plot(sim.k_list, mat_init[:,1,1], label='1,1')
    plt.legend()
    plt.show()
    deriv = sim.get_k_partial(sim.mat_init)
    plt.plot(sim.k_list, deriv[:, 0, 0], label='0,0')
    plt.plot(sim.k_list, deriv[:, 1, 0], label='1,0')
    plt.plot(sim.k_list, deriv[:, 0, 1], label='0,1')
    plt.plot(sim.k_list, deriv[:, 1, 1], label='1,1')
    plt.legend()
    plt.show()

def plot_H_const():
    sim = Simulation(t_end=30, n_steps=1000)
    sim.define_pulse(sigma=3, lam=774, t_start=11, E0=1)
    sim.define_system(num_k=100, a=9.8) 
    sim.define_bands(Ec=4, Ev=-3, tc=-1.5, tv=0.5)
    sim.set_H_constant(dipole_element=1e11)
    h_const = sim.h_const
    # plt.plot(sim.k_list, h_const[:,0,0], label='0,0')
    plt.plot(sim.k_list, h_const[:,1,0], label='1,0')
    plt.plot(sim.k_list, h_const[:,0,1], label='0,1')
    # plt.plot(sim.k_list, h_const[:,1,1], label='1,1')
    plt.legend()
    plt.show()

def plot_H_deriv():
    sim = Simulation(t_end=30, n_steps=1000)
    sim.define_pulse(sigma=3, lam=774, t_start=11, E0=1)
    sim.define_system(num_k=100, a=9.8) 
    sim.define_bands(Ec=4, Ev=-3, tc=-1.5, tv=0.5)
    sim.set_H_constant(dipole_element=9e-29)
    h_const = sim.h_const 
    deriv = sim.get_k_partial(sim.h_const)
    plt.plot(sim.k_list, deriv[:,0,0], label='0,0')
    plt.plot(sim.k_list, deriv[:,1,0], label='1,0')
    plt.plot(sim.k_list, deriv[:,0,1], label='0,1')
    plt.plot(sim.k_list, deriv[:,1,1], label='1,1')
    plt.legend()
    plt.show()
    
def plot_E_field():
    sim = Simulation(t_end=30, n_steps=1000)
    sim.define_pulse(sigma=3, lam=774, t_start=11, E0=1)
    sim.define_system(num_k=100, a=9.8) 
    sim.define_bands(Ec=4, Ev=-3, tc=-1.5, tv=0.5)
    sim.set_H_constant(dipole_element=9e-29)
    time = sim.time
    E_mat = np.zeros((len(sim.time), sim.num_k, 2,2))
    for i,t in enumerate(sim.time):
        E_mat[i] = np.reshape(sim.get_E(t),(sim.num_k, 2, 2))
    plt.plot(time, E_mat[:,0,0,1]) 
    plt.show()

def check_commutator():
    sim = Simulation(t_end=30, n_steps=100)
    sim.define_pulse(sigma=3, lam=774, t_start=11, E0=0) # E_null = 0 -> no field, rho and H diagonal, commute
    sim.define_system(num_k=50, a=9.8) 
    sim.define_bands(Ec=4, Ev=-3, tc=-1.5, tv=0.5)
    sim.set_H_constant(dipole_element=9e-29)
    rho = sim.mat_init
    commutator = np.reshape(sim.commute(rho, t=500), (sim.num_k, 2, 2)) #check for H at random time 
    print(np.sum(np.abs(commutator)))

def check_rho_hermitian():
    sim = Simulation(t_end=30, n_steps=1000)
    sim.define_pulse(sigma=3, lam=774, t_start=11, E0=0) # E_null = 0 -> no field, rho and H diagonal, commute
    sim.define_system(num_k=100, a=9.8) 
    sim.define_bands(Ec=4, Ev=-3, tc=-1.5, tv=0.5)
    sim.set_H_constant(dipole_element=9e-29)
    sim.integrate()
    hermitian_mask = np.vectorize(ishermitian, signature='(i,j)->()')(sim.solution)
    all_hermitian = hermitian_mask.all()
    print(all_hermitian)

def check_trace_const():
    sim = Simulation(t_end=100, n_steps=5000)
    sim.define_pulse(sigma=5, lam=774, t_start=50, E0=1e9) #E_0 = 1e11 roundabout corresponding to I = 1.5e14 W/cm^2
    sim.define_system(num_k=100, a=9.8, T2=20) 
    sim.define_bands(Ec=4, Ev=-3, tc=-1.5, tv=0.5)
    sim.set_H_constant(dipole_element=9e-29) # 9e-29 corresponds to roundabout 9 a.u.
    sim.integrate() 
    sim.get_current()
    results = Plot(sim)
    results.plot_current()
    # sim.pulse.get_vector_potential()
    # results.plot_field_A()
    results.get_heatmap_rho()
    # rho = np.abs(sim.solution)
    rho =  sim.solution
    traces = np.einsum('ijkk -> ij', rho)
    print(traces)
    print(np.allclose(traces, traces[0,0]))

def check_h_partial():
    sim = Simulation(t_end=30, n_steps=1000)
    sim.define_pulse(sigma=3, lam=774, t_start=11, E0=0) # E_null = 0 -> no field, rho and H diagonal, commute
    sim.define_system(num_k=100, a=9.8) 
    sim.define_bands(Ec=4, Ev=-3, tc=-1.5, tv=0.5)
    H_partial = sim.get_H_partial()
    plt.plot(sim.k_list, H_partial[:,0,0], label='0,0')
    plt.plot(sim.k_list, H_partial[:,1,0], label='1,0')
    plt.plot(sim.k_list, H_partial[:,0,1], label='0,1')
    plt.plot(sim.k_list, H_partial[:,1,1], label='1,1')
    plt.legend()
    plt.show()

def plot_dipole_mat():
    sim = Simulation(t_end=30, n_steps=1000)
    sim.define_pulse(sigma=3, lam=774, t_start=11, E0=0) # E_null = 0 -> no field, rho and H diagonal, commute
    sim.define_system(num_k=1000, a=9.8) 
    sim.define_bands(Ec=4, Ev=-3, tc=-1.5, tv=0.5)
    sim.set_H_constant(dipole_element=9e-29)
    plt.plot(sim.k_list, sim.dipole_mat[:,0,0], label='0,0')
    plt.plot(sim.k_list, np.real(sim.dipole_mat[:,1,0]), label='re(1,0)')
    plt.plot(sim.k_list, np.imag(sim.dipole_mat[:,1,0]), label='im(1,0)')
    plt.plot(sim.k_list, sim.dipole_mat[:,0,1], label='0,1')
    plt.plot(sim.k_list, np.imag(sim.dipole_mat[:,0,1]), label='im(0,1)')
    plt.plot(sim.k_list, np.real(sim.dipole_mat[:,1,1]), label='re(1,1)')
    plt.legend()
    plt.show()

check_trace_const()