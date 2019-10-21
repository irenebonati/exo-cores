import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd

from scipy.optimize import curve_fit # for fitting the density with a function


MEarth = 5.972e24 #kge
R_Earth = 6371 #km
G = 6.67384e-11 #m3/kg/s2


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
    return rho_0.tolist(), L_rho.tolist(), A_rho.tolist()

def name_file(XFe, Mp, FeM):
    return "data_prof_M_ {:.1f}_Fe_{:.0f}.0000_FeM_{:2.0f}.0000.res".format(Mp, XFe, FeM)

def Earth():
        """ Default values for Earth (see Labrosse 2015) """
        param = {
                    "r_OC" : 3480.e3,          # Core radius (m)
                    "r_IC" : 1221.e3,          # Present inner core radius (m)
                    "r_IC_0": 0., # Initial inner core radius (m)
                    #
                    "alpha_c" : 1.3e-5,           # Thermal expansion coefficient at center (K-1)
                    "CP" : 750.,             # Specific heat (J.kg-1.K-1)
                    "gamma" : 1.5,              # Grueneisen parameter
                    # Parameters obtained from the fitting of the density
                    "rho_0" :  12502.,           # Density at center (kgm-3)
                    "L_rho" : 8039e3,             # (m)
                    "A_rho" : 0.484,              # (no unit)
                    # Parameter obtained from the qs_**.res  
                    "Q_CMB" : 7.4e12,           # Present CMB flux (W)
                    # Parameters not expected to be modified
                    "Deltarho_ICB" : 500.,             # Density jump at ICB (kgm-3)
                    "DeltaS" : 127.,             # Entropy of crystallization (Jkg-1K-1)
                    "H" : 0.,               # Radioactivity
                    "k_c" : 163.,             # Thermal conductivity at center (Wm-1K-1)
                    "beta" : 0.83,           # coefficient of compositional expansion
                    "chi0" : 0.056,            # Difference in mass fraction of light elements across ICB
                    # Phase diag (maybe not modified?)
                    "dTL_dchi" : -21.e3,           # Compositional dependence of liquidus temperature (K)
                    "dTL_dP" : 9.e-9,            # Pressure dependence of liquidus temperature (Pa-1)
                    "TL0" : 5700.,            # Melting temperature at center (K)
        }
        return param

def average_volume(profile, name_variable):
    """ Average of the parameter called name_variable over the full volume """
    profile["dV"] = 4*np.pi*profile["r(m)"]
    volume = profile["dV"].sum()
    quantity = profile["dV"]*profile[name_variable]
    quantity_total = quantity.sum()
    return quantity_total/volume

def write_parameter_file(XFe, Mp, FeM):
    """ write the yaml parameter file  """
    filename = name_file(XFe, Mp, FeM)
    core = open_data_profiles(filename, core=True)

    # initialisation with Earth parameters
    param = Earth()

    # modification of the parameters from the profiles
    param["rho_0"], param["L_rho"], param["A_rho"] = find_Lrho_Arho(core)
    param["r_OC"] = core["r(m)"].iloc[-1].tolist()
    param["CP"] = average_volume(core, "Cp(J/kgK)").tolist()
    param["alpha_c"] = average_volume(core, "alpha(10^-5 1/s)").tolist() * 1e-5
    param["gamma"] = average_volume(core, "Gruneisen(1)").tolist()
    output_filename = "M_ {:.1f}_Fe_{:.0f}.0000_FeM_{:2.0f}.0000.yaml".format(Mp, XFe, FeM)

    # write yaml file
    with open(output_filename, 'w') as outfile:
        yaml.dump(param, outfile, default_flow_style=False)


write_parameter_file(30, 1.2, 0)