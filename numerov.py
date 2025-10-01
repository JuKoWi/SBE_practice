import numpy as np


# Solving the SE for simple 1D potentials
# The method is adopted from
# https://www.fisica.uniud.it/%7Egiannozz/Didattica/MQ/Software/C/harmonic1.c
def numerov_forward(u0, um, F0, Fm, Fp, h):
    """
    u0 = u[i], um = u[i-1]
    F0 = F[i], Fm = F[i-1], Fp = F[i+1]
    """
    up = 2 * u0 - um - (h**2 / 12) * (10 * F0 * u0 + Fm * um)
    up /= 1 + (h**2 / 12) * Fp
    return up


def numerov_backward(u0, up, F0, Fm, Fp, h):
    """
    u0 = u[i], up = u[i+1]
    F0 = F[i], Fm = F[i-1], Fp = F[i+1]
    """
    um = 2 * u0 - up - (h**2 / 12) * (10 * F0 * u0 + Fp * up)
    um /= 1 + (h**2 / 12) * Fm
    return um


def numerov_solve_out(u, F, h, gerade=True, imax=-1):
    N = imax
    if imax == -1:
        N = len(u)

    if gerade:
        u[0] = 1
        u[1] = (1 - 5 * h**2 / 12 * F[0]) / (1 + h**2 / 12 * F[1]) * u[0]
    else:
        u[0] = 0
        u[1] = h

    n_nodes = 0
    for i in range(1, N - 1):
        u[i + 1] = numerov_forward(u[i], u[i - 1], F[i], F[i - 1], F[i + 1], h)
        if u[i + 1] * u[i] < 0:
            n_nodes += 1
    return u, n_nodes


def numerov_solve_in(u, F, h, imin=0):
    N = len(u)

    u[N - 1] = h
    u[N - 2] = 2 * (1 - 5 * h**2 / 12 * F[N - 1]) / (1 + h**2 / 12 * F[N - 2]) * u[N - 1]

    for i in range(N - 2, imin, -1):
        u[i - 1] = numerov_backward(u[i], u[i + 1], F[i], F[i - 1], F[i + 1], h)
    return u


def solve_schroedinger(V, k, gerade, h, Etry=0, maxiter=100):
    u = np.zeros_like(V)
    N = len(u)
    Elower = V.min()
    Eupper = V.max()

    niter = 0
    while niter < maxiter:  # and (np.abs(deltaE) > Etol):
        F = 2 * (Etry - V)

        icl = -1  # Position of the outermost classical turning point
        for i in range(N - 1):
            if F[i] * F[i + 1] < 0:
                icl = i + 1

        u = np.zeros_like(V)
        # Outward integration and count of nodes
        u, n_nodes = numerov_solve_out(u, F, h, gerade=gerade, imax=icl + 1)
        ucl = u[icl]

        if n_nodes != k:
            # Adjust energy bounds if number of nodes is not correct
            if n_nodes > k:
                Eupper = Etry
            else:
                Elower = Etry
            Etry = Elower + (Eupper - Elower) / 2

        # If number of nodes is ok, proceed with inward integration
        if n_nodes == k:
            u = numerov_solve_in(u, F, h, imin=icl)
            ucl /= u[icl]
            u[icl:] *= ucl

            djmp = (u[icl + 1] + u[icl - 1] - (2 - h**2 * F[icl]) * u[icl]) / h
            if djmp * u[icl] > 0:
                Eupper = Etry
            else:
                Elower = Etry
            Etry = Elower + (Eupper - Elower) / 2

        niter += 1
    return u, Etry, Eupper - Elower, n_nodes


def second_deriv(u, h):
    N = len(u)
    ddu = np.zeros_like(u)

    for i in range(1, N - 1):
        ddu[i] = (u[i + 1] + u[i - 1] - 2 * u[i]) / h**2
    return ddu


def symmetric(u, gerade=True):
    if gerade:
        psi = np.concatenate([np.flip(u), u[1:]])
    else:
        psi = np.concatenate([-np.flip(u), u[1:]])
    return psi


### RADIAL SCHROEDINGER EQUATION


def numerov_solve_out_radial(u, F, F0u0, ell, h, imax=-1):
    N = imax
    if imax == -1:
        N = len(u)

    u[0] = 0
    u[1] = h ** (ell + 1)
    u[2] = 2 * u[1] - u[0] - (h**2 / 12) * (10 * F[1] * u[1] + F0u0)
    u[2] /= 1 + (h**2 / 12) * F[2]

    n_nodes = 0
    for i in range(2, N - 1):
        u[i + 1] = numerov_forward(u[i], u[i - 1], F[i], F[i - 1], F[i + 1], h)
        if u[i + 1] * u[i] < 0:
            n_nodes += 1
    return u, n_nodes


def solve_schroedinger_radial(V, ell, k, h, Etry=0, maxiter=100):
    u = np.zeros_like(V)
    N = len(u)

    rs = h * np.arange(0, N)
    rs[0] = 1e-6

    # TODO: Implement for general potential
    if ell == 0:
        F0u0 = -2 * (-1)  # lim_h->0 h*V(h) for Z = 1
    elif ell == 1:
        F0u0 = -2
    else:
        F0u0 = 0

    Elower = V.min()
    Eupper = V.max()

    niter = 0
    while niter < maxiter:  # and (np.abs(deltaE) > Etol):
        F = 2 * (Etry - V) - ell * (ell + 1) / rs**2

        icl = -1  # Position of the outermost classical turning point
        for i in range(N - 1):
            if F[i] * F[i + 1] < 0:
                icl = i + 1
        u = np.zeros_like(V)

        # Outward integration and count of nodes
        u, n_nodes = numerov_solve_out_radial(u, F, F0u0, ell, h, imax=icl + 1)
        ucl = u[icl]

        if n_nodes != k:
            # Adjust energy bounds if number of nodes is not correct
            if n_nodes > k:
                Eupper = Etry
            else:
                Elower = Etry
            Etry = Elower + (Eupper - Elower) / 2

        # If number of nodes is ok, proceed with inward integration
        if n_nodes == k:
            u = numerov_solve_in(u, F, h, imin=icl)
            ucl /= u[icl]
            u[icl:] *= ucl

            djmp = (u[icl + 1] + u[icl - 1] - (2 - h**2 * F[icl]) * u[icl]) / h
            if djmp * u[icl] > 0:
                Eupper = Etry
            else:
                Elower = Etry
            Etry = Elower + (Eupper - Elower) / 2

        niter += 1
    return u, Etry, Eupper - Elower, n_nodes
