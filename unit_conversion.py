import scipy.constants as constants
import numpy as np

def eV_to_au(eV):
    return eV / constants.physical_constants['Hartree energy in eV'][0]


def au_to_ev(au):
    return au *constants.physical_constants['Hartree energy in eV'][0]
    
def angstrom_to_bohr(val):
    bohr_radius_in_meters = constants.physical_constants['Bohr radius'][0]  # in meters
    angstrom_in_meters = constants.angstrom  # 0 Å = 1e-10 m
    return val * angstrom_in_meters / bohr_radius_in_meters

def bohr_to_angstrom(val):
    bohr_radius_in_meters = constants.physical_constants['Bohr radius'][0]  # in meters
    angstrom_in_meters = constants.angstrom
    return val * bohr_radius_in_meters / angstrom_in_meters

def nm_to_au(wavelength_nm):
    bohr_radius_m = constants.physical_constants['Bohr radius'][0]  # in meters
    au = wavelength_nm * constants.nano / bohr_radius_m
    return au

def lam_to_omega(lam_au):
    c_au = 137.035999084  # speed of light in atomic units
    omega_au = 2*np.pi * c_au /lam_au
    return omega_au

def fs_to_au(fs):
    s = fs * constants.femto
    au = s / constants.physical_constants['atomic unit of time'][0]
    return au

def au_to_fs(au):
    return (au * constants.physical_constants['atomic unit of time'][0]) /constants.femto

def Vpm_to_au(Vpm):
    return Vpm / constants.physical_constants['atomic unit of electric field'][0]

def au_to_Vpm(au):
    return au * constants.physical_constants['atomic unit of electric field'][0]

def Cm_to_au(Cm):
    return Cm / constants.physical_constants['atomic unit of electric dipole moment'][0]
