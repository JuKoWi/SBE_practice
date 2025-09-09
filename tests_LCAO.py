from SBE_on_LCAO import Simulation, Plot
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

def check_commutator():
    sim = Simulation(t_end=100, n_steps=5000)
    sim.define_pulse(sigma=5, lam=740, t_start=50, E0=1e10) #E_0 = 1e11 roundabout corresponding to I = 1.5e14 W/cm^2
    sim.use_LCAO(num_k=100, a=1.32, scale_H=1, m_max=4)
    sim.integrate() 
    rho = sim.mat_init
    commutator = np.reshape(sim.commute(rho, t=0), (sim.num_k, 4, 4)) #check for H at random time 
    print(np.sum(np.abs(commutator)))

def check_rho_hermitian():
    sim = Simulation(t_end=100, n_steps=5000)
    sim.define_pulse(sigma=5, lam=740, t_start=50, E0=1e10) #E_0 = 1e11 roundabout corresponding to I = 1.5e14 W/cm^2
    sim.use_LCAO(num_k=100, a=1.32, scale_H=1, m_max=4)
    sim.integrate() 
    hermitian_mask = np.vectorize(ishermitian, signature='(i,j)->()')(sim.solution)
    all_hermitian = hermitian_mask.all()
    print(all_hermitian)

def check_trace_const():
    sim = Simulation(t_end=100, n_steps=5000)
    sim.define_pulse(sigma=5, lam=740, t_start=50, E0=1e10) #E_0 = 1e11 roundabout corresponding to I = 1.5e14 W/cm^2
    sim.use_LCAO(num_k=100, a=1.32, scale_H=1, m_max=4)
    sim.integrate() 
    results = Plot(sim)
    results.get_heatmap_rho()
    rho =  sim.solution
    traces = np.einsum('ijkk -> ij', rho)
    print(traces)
    print(np.allclose(traces, traces[0,0]))


check_trace_const()
check_rho_hermitian()
check_commutator()