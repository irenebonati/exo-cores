import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd

from scipy.optimize import curve_fit # for fitting the density with a function


def open_data_profiles(filename, core=False): 
    names = ["g(m/s^2)", "p(GPa)", "rho(kg/m^3)","r(m)", "T(K)", "oups", "Cp(J/kgK)", "alpha(10^-5 1/s)", "Gruneisen(1)", \
             "KT(GPa)", "KS(GPa)", "G(GPa)", "ElCond (Siemens)", "Material-Parameter" ]
    data = pd.read_csv(filename, skipinitialspace=True, sep=" ", 
                       names=names, index_col=False)
    if core == True:
        data = data[data["Material-Parameter"]==8.]
    return data

    
def density_Labrosse2015(r, *args):
    """ Equation (5) in Labrosse 2015 """
    rho_0, L_rho, A_rho = args
    return rho_0*(1-r**2/L_rho**2-A_rho*r**4/L_rho**4)

def find_Lrho_Arho(profile):
    """ least square to fit the function density to the data.count
    
    curve_fit use non-linear least square,
    the default method is Levenberg-Marquardt algorithm as implemented in MINPACK.
    (doesn't handle bounds and sparse jacobians. Should be enough here)
     """
    rho = profile["rho(kg/m^3)"]
    radius = profile["r(m)"]
    initial_guess = 12500, 8000e3, 0.484 # initial values from Labrosse 2015 (for the Earth)
    popt, pcov = curve_fit(density_Labrosse2015, radius, rho, initial_guess)
    rho_0, L_rho, A_rho = popt
    return rho_0, L_rho, A_rho


def write_parameter_file(param):
    
    def Earth():   
        param = {  
                    "alpha_c" = 1.3e-5,           # Thermal expansion coefficient at center (K-1)
                    "r_OC" = 3480.e3,          # Core radius (m)
                    "r_IC" = 1221.e3,          # Present inner core radius (m) --> same as c
                    "CP" = 750.,             # Specific heat (Jkg-1K-1)
                    "Deltarho_ICB" = 500.,             # Density jump at ICB (kgm-3)
                    "DeltaS" = 127.,             # Entropy of crystallization (Jkg-1K-1)
                    "DTS_DTAD" = 1.65,             # Adiabatic temperature gradient (K/m)
                    "gamma" = 1.5,              # Grueneisen parameter
                    "GC" = 6.67384e-11,     # Gravitational constant (m3kg-1s-2)
                    "H" = 0.,               # Radioactivity
                    "K_c" = 1403.e9,         # Bulk modulus at center (Pa) (Labrosse+2015)
                    "Kprime_0" = 3.567,            # unitless
                    "k_c" = 163.,             # Thermal conductivity at center (Wm-1K-1)
                    #L_T           :  6042.e3          # Temperature length scale (m)
                    "Q_CMB" = 7.4e12,           # Present CMB flux (W)
                    "rho_c" =  12502.,           # Density at center (kgm-3)
                    "rho_oc" = 10.e3,            # Density outer core (kgm-3)
                    "rho_0" = 7.5e3,            # Density at 0 pressure (kgm-3)
                    "T_s0" = 5270.,            # Solidification temperature at center (K,Labrosse 2001)
                    "Ts_ICB" = 5600.,            # Present temperature at ICB (K)
                    "beta" = 0.83,
                    "dTL_dchi" = -21.e3,           # Compositional dependence of liquidus temperature (K)
                    "dTL_dP" = 9.E-9,            # Pressure dependence of liquidus temperature (Pa-1)
                    "chi0" = 0.056,            # Difference in mass fraction of light elements across ICB
                    "TL0" = 5700.,            # Melting temperature at center (K)
        }
        return param