
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.linalg import eig
import numpy as np

# Import the prepared functions
from numerov import solve_schroedinger, symmetric



def poeschl_teller(xs, lam=5, a=1):
    return -lam * (lam + 1) * a**2 / (2 * np.cosh(a * xs) ** 2)



# Input parameters
N_single = 40000  # Number of grid points
xmax_single = 40 # Extent of the grid
lam = 5

# Preparation of the grid
xg = np.linspace(0, xmax_single, N_single)
h = xg[1] - xg[0]

# ...and the potential
V = poeschl_teller(xg, lam=lam, a=0.5)

# why do only negative test energies give the right energies?

def create_wf(k, gerade, V, Etry=-1):
    u, E, dE, n_nodes = solve_schroedinger(V, k=k, gerade=gerade, h=h, Etry=Etry)
    print(n_nodes)
    print(E)
    psi0 = symmetric(u, gerade=gerade)
    psi0 /= np.sqrt(np.trapezoid(psi0 * psi0, dx=h))
    return psi0

# ao = np.zeros((lam, 2*N_single+1))
# print(np.shape(ao))
k = 0
ao = np.zeros((lam, N_single * 2 - 1)) 
for i in range(lam): 
    gerade = (i%2 == 0)
    ao[i] = create_wf(k, gerade, V)
    if i%2 == 1:
        k +=1
print(np.shape(ao))
np.save("numerov-five.npy", arr=ao)



V_plot = symmetric(V, gerade=True)
xg = symmetric(xg, gerade=False)
np.save("numerov-grid.npy", xg)

fig, axs = plt.subplots(2, 1, sharex=False)

axs[0].plot(xg, V_plot)

for i,orb in enumerate(ao):
    axs[1].plot(xg, orb, label=f"psi{i} with E = {orb[1]:.2f}")

axs[0].set_xlim((-5, 5))
axs[1].set_xlim((-6, 6))

plt.legend(loc='upper right')
plt.show()
