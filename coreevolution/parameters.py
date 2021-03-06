import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import glob, os

from scipy.optimize import curve_fit, minimize_scalar # for fitting the density with a function


MEarth = 5.972e24 #kg
R_Earth = 6371 #km
G = 6.67384e-11 #m3/kg/s2

def name_file(XFe, Mp, FeM):
    return "data_prof_M_ {:.1f}_Fe_{:.0f}.0000_FeM_{:2.0f}.0000.res".format(Mp, XFe, FeM)

def read_data_profiles(filename, core=False):
    names = ["g(m/s^2)", "p(GPa)", "rho(kg/m^3)","r(m)", "T(K)", "oups", "Cp(J/kgK)", "alpha(10^-5 1/s)", "Gruneisen(1)","KT(GPa)", "KS(GPa)", "G(GPa)", "ElCond (Siemens)", "Material-Parameter" ]
    data = pd.read_csv(filename, skipinitialspace=True, sep=" ", names=names, index_col=False)
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
    initial_guess = 12502, 8039e3, 0.484 # initial values from Labrosse 2015 (for the Earth)
    popt, pcov = curve_fit(density_Labrosse2015, radius, rho, initial_guess)
    rho_0, L_rho, A_rho = popt
    return rho_0.tolist(), L_rho.tolist(), A_rho.tolist()

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
                    "rho_c" :  12502.,           # Density at center (kgm-3)
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
    """ Average the variable name_variable over the volume """
    profile["dV"] = 4*np.pi*profile["r(m)"]
    volume = profile["dV"].sum()
    quantity = profile["dV"]*profile[name_variable]
    quantity_total = quantity.sum()
    return quantity_total/volume

def T_liquidus_core(P, S=0):
    """ T_{\rm melt} = 6500 * (p/340)^{0.515} / (1 - ln(1-X_{\rm S}) ) """
    return 6500.*(P/340)**0.515/(1-np.log(1-S))

def T_liquidus_mantle(P, FeM):
    """ 5400 * (p/140)^{0.48} / (1 - ln(1-\#Fe_{\rm M}) ) """
    return 5400.*(P/140)**0.48/(1-np.log(1-FeM))

def T_adiabat(radius, Lrho, Arho, T0, gamma):
    return T0*(1-radius**2/Lrho**2-Arho*radius**4/Lrho**4)**gamma

def gravity(r, *args):
    G = 6.67430e-11
    rho_0, L_rho, A_rho = args
    parenthesis = 1-3/5*r**2/L_rho**2-3/7*A_rho*r**4/L_rho**4
    return 4.*np.pi/3.*G*rho_0*r*parenthesis

def pressure_diff(r, *args):  #in GPa
    rho_0, L_rho, A_rho = args
    G = 6.67430e-11
    K0 = L_rho**2/3.*2.*np.pi*G*rho_0**2 /1e9 #in GPa
    parenthesis = r**2/L_rho**2-4./5.*r**4/L_rho**4
    return -K0*parenthesis

def find_r_IC_adiabat(rho_0, Lrho, Arho, P0, T0, gamma, S=0):
    def Delta_T(radius):
        P = pressure_diff(radius, rho_0, Lrho, Arho)+P0
        Ta = T_adiabat(radius, Lrho, Arho, T0, gamma)
        TL = T_liquidus_core(P, S)
        return (Ta - TL)**2
    res = minimize_scalar(Delta_T, bounds=(0., 6e6), method='bounded') 
    r_IC = res.x
    if r_IC < 1: r_IC = np.array(0.)
    return r_IC.tolist()

def find_CMB(profiles):
    core = profiles[profiles["Material-Parameter"]==8.]
    #print(core)
    index_max = core["r(m)"].idxmax()
    return index_max, core["r(m)"].iloc[0], core["p(GPa)"].iloc[0], core["T(K)"].iloc[0]
    
def center(profiles): 
    return profiles["r(m)"].iloc[-1], profiles["p(GPa)"].iloc[-1], profiles["T(K)"].iloc[-1]

def figure(data, ax, symb="-"): 
    ax[0,0].plot(data["r(m)"]/1e3, data["T(K)"], symb)
    ax[0,1].plot(data["r(m)"]/1e3, data["g(m/s^2)"], symb)
    ax[1,0].plot(data["r(m)"]/1e3, data["rho(kg/m^3)"], symb)
    ax[1,1].plot(data["r(m)"]/1e3, data["p(GPa)"], symb)
    ax[0,0].set_ylabel("Temperature(K)")
    ax[0,1].set_ylabel("g (m/s$^2$)")
    ax[1,0].set_ylabel("Density (kg/m$^3$)")
    ax[1,1].set_ylabel("Pressure (GPa)")
    ax[1,0].set_xlabel("Radius (km)")
    ax[1,1].set_xlabel("Radius (km)")


def calculate_parameters(filename, verbose=False):
    data = read_data_profiles(filename)
    core = data[data["Material-Parameter"]==8.]
    # core = read_data_profiles(filename, core=True)
    # print(core)
    # extract the mass, XFe, FeM
    newstr = ''.join((ch if ch in '0123456789.' else ' ') for ch in os.path.basename(filename[:-4]))
    Mp, XFe, FeM = [float(i) for i in newstr.split()]
    # initialize parameters with Earth
    param = Earth()
    #update parameters
    param["r_planet"] = data["r(m)"].iloc[0].tolist()
    param["Mp"], param["XFe"], param["FeM"] = Mp, XFe, FeM
    param["rho_0"], param["L_rho"], param["A_rho"] = find_Lrho_Arho(core) #rho_0 is the density at the center
    param["CP"] = average_volume(core, "Cp(J/kgK)").tolist()  #average Cp
    param["gc"] = average_volume(core, "g(m/s^2)").tolist()  #average gc
    param["alpha_c"] = average_volume(core, "alpha(10^-5 1/s)").tolist() * 1e-5
    param["gamma"] = average_volume(core, "Gruneisen(1)").tolist()
    param["T0"] = core["T(K)"].iloc[-1].tolist() #at the center
    param["P0"] = core["p(GPa)"].iloc[-1].tolist() #at the center
    param["Tcmb"] = core["T(K)"].iloc[0].tolist() #at the CMB
    param["Pcmb"] = core["p(GPa)"].iloc[0].tolist() #at the CMB
    P0 = core["p(GPa)"].iloc[-1]
    param["r_IC_0"] = find_r_IC_adiabat(param["rho_0"], param["L_rho"], param["A_rho"], P0, param["T0"], param["gamma"], S=0.)
    param["r_IC_005"] = find_r_IC_adiabat(param["rho_0"], param["L_rho"], param["A_rho"], P0, param["T0"], param["gamma"], S=0.05)
    param["r_IC_011"] = find_r_IC_adiabat(param["rho_0"], param["L_rho"], param["A_rho"], P0, param["T0"], param["gamma"], S=0.11)
    param["r_OC"] = core["r(m)"].iloc[0].tolist()
    param["TL0_0"] = T_liquidus_core(P0, S=0).tolist()
    param["TL0_005"] = T_liquidus_core(P0, S=0.05).tolist()
    param["TL0_011"] = T_liquidus_core(P0, S=0.11).tolist()
    param["K_c"] = 1403.e9 # Earth's bulk modulus at the center (Labrosse+2015)
    return param, core

def write_parameter_file(filename, fig=False, folder="", verbose=False):
    """ Write the yaml file including all the parameters """
    param, core = calculate_parameters(filename)
    # newstr = ''.join((ch if ch in '0123456789.' else ' ') for ch in os.path.basename(filename[:-4]))
    Mp, XFe, FeM = param["Mp"], param["XFe"], param["FeM"]
    if verbose: print (Mp,XFe,FeM)
    #output_filename = filename[:-4]+".yaml"
    output_filename = folder+"M_ {:.1f}_Fe_{:.0f}.0000_FeM_{:2.0f}.0000.yaml".format(Mp, XFe, FeM)
    # create the yaml parameter file
    with open(output_filename, 'w') as outfile:
        yaml.dump(param, outfile, default_flow_style=False)
    # if necessary, plot the figure to check the fits and values
    if fig:
        #rho = core["rho(kg/m^3)"]
        radius = core["r(m)"]
        P0 = core["p(GPa)"].iloc[-1]
        fig, ax3 = plt.subplots(2,2)
        figure(core, ax3)
        ax3[1,0].plot(radius[::100]/1e3, density_Labrosse2015(radius[::100], param["rho_0"], param["L_rho"], param["A_rho"]), '+')
        ax3[0,1].plot(radius[::100]/1e3, gravity(radius[::100], param["rho_0"], param["L_rho"], param["A_rho"]), '+')
        ax3[1,1].plot(radius[::100]/1e3, pressure_diff(radius[::100], param["rho_0"], param["L_rho"], param["A_rho"])+P0, '+')
        ax3[0,0].plot(radius[::100]/1e3, T_adiabat(radius[::100], param["L_rho"], param["A_rho"], param["T0"], param["gamma"]), '+', label="fit")
        ax3[0,0].plot(radius/1e3, T_liquidus_core(pressure_diff(radius, param["rho_0"], param["L_rho"], param["A_rho"])+P0), label="Melting T")
        ax3[0,0].plot(np.array([param["r_IC_0"], param["r_IC_0"]])/1e3, [core["T(K)"].iloc[-1], core["T(K)"].iloc[0]])
        ax3[0,0].legend()
    return param #param["Mp"], param["XFe"], param["FeM"], param["rho_0"], param["L_rho"], param["A_rho"]


def explore_all_create_yaml(folder, fig=False, verbose=False):
    files = [f for f in glob.glob(folder + "data_prof*.res")]
    all_files = folder+"/all_files_list.txt"
    print("Creating {} data file".format(all_files))
    for file in files: 
        if file.split('/')[-1] != "data_IS.res":
            param = write_parameter_file(file, folder=folder)
            Mp, XFe, FeM, rho, L, A = param["Mp"], param["XFe"], param["FeM"], param["rho_0"], param["L_rho"], param["A_rho"]
            with open(all_files, 'a+') as the_file:
                the_file.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(Mp, XFe, FeM, rho, L, A,
                                                                  param["r_IC_0"], param["r_IC_005"], param["r_IC_011"],
                                                                  param["r_OC"], param["r_planet"],
                                                                  param["Pcmb"], param["Tcmb"],
                                                                  param["P0"], param["T0"]))
    if fig:  #TODO to be changed (the names are not correct)
        all_files = "all_files_list.txt"
        names = ["Mp", "XFe", "FeM", "rho", "L", "A", "r_ic", "r_ic_005"]
        data = pd.read_csv(all_files, skipinitialspace=True, sep=" ", names=names)
        data = data[data["FeM"]==0.]
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        sc = ax[0].tricontourf(data["Mp"], data["XFe"], data["r_ic"]/1e3)
        plt.colorbar(sc, ax=ax[0])
        sc = ax[1].tricontourf(data["Mp"], data["XFe"], data["rho"])
        plt.colorbar(sc, ax=ax[1])
        ax[0].set_ylabel("XFe")
        ax[0].set_xlabel("Mass planet")
        ax[0].set_title("L density")
        ax[1].set_xlabel("Mass planet")
        ax[1].set_title("density at center")

    
if __name__ == "__main__":
#    filename = name_file(25, 0.8, 10)
#    write_parameter_file("Ini_With_DTcmb/"+ filename, fig=True)
    explore_all_create_yaml("Ini_warm/", fig=False)
    plt.show()
    