import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import scipy.integrate as integrate
import yaml

year = 365.25*3600*24 #s

# Define class for test case
class Rocky_Planet():
    """ Rocky planet, defines physical parameters and evolution """

    def __init__(self):
        self.parameters()

    def parameters(self):
        pass # parameters are defined in class

    def A_rho(self):
        return (5. * self.Kprime_0 - 13.) / 10.
    
    def evolution(self):
        evolution = Evolution(self)
        evolution.energies()
        evolution.profiles()

    def read_parameters(self, file):
        with open(file, 'r') as stream:
            try:
                dict_param = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        for k, v in dict_param.items():
            setattr(self, k, float(v))

class Evolution():
    """ Calculate the thermal (also chemical?) evolution, associated to one planet """

    def __init__(self, planet):
        self.planet = planet
        
    def calc_L_T(self):
        """ Temperature change length scale """
        return np.sqrt(3.*self.planet.CP/(2.*np.pi*self.planet.alpha_c*self.planet.rho_c*self.planets.GC))
        
    def calc_L_rho(self):
        """ Density change length scale """
        return np.sqrt(3.*self.planet.K_c/(2.*np.pi*G*self.planet.rho_0*self.planet.rho_c)*(np.log(self.planet.rho_c/self.planet.rho_0)+1.))

    def dTL_dr_IC(self, r):
        ''' Melting temperature jump at ICB '''
        return -self.planet.K_c * 2.*self.planet.dTL_dP * r / self.planet.L_rho**2. \
            + 3. * self.planet.dTL_dchi * self.planet.chi0 * r**2. / (self.planet.L_rho**3. * self.fC(self.planet.r_OC / self.planet.L_rho, 0.))

    def fC(self, r, delta): 
        '''fC (Eq. A1 Labrosse 2015)'''
        return r**3. * (1 - 3. / 5. * (delta + 1) * r**2.- 3. / 14. * (delta + 1) \
            * (2 * self.planet.A_rho - delta) * r**4.)

    def fX(self, r, r_IC):
        '''fX (Eq. A15 Labrosse 2015)'''
        return (r)**3. * (-r_IC**2. / 3. / self.planet.L_rho**2. + 1./5. * (1.+r_IC**2./self.planet.L_rho**2.) \
                *(r)**2.-13./70. * (r)**4.) 

    def rho(self, r):
        ''' Density (Eq. 5 Labrosse 2015)'''
        return self.planet.rho_c * (1. - r**2. / self.planet.L_rho**2. - self.planet.A_rho * r**4. / self.planet.L_rho**4.)

    def T_melt(self, r):
        ''' Melting temperature at ICB (Eq. 14 Labrosse 2015)'''
        return self.planet.TL0 - self.planet.K_c * self.planet.dTL_dP * r**2. / self.planet.L_rho**2. + self.planet.dTL_dchi * self.planet.chi0 * r**3. \
                / (self.planet.L_rho**3. * self.fC(self.planet.r_OC / self.planet.L_rho, 0.))

    def PL(self, r):
        '''Latent heat power'''
        return 4. * np.pi * r**2. * self.T_melt(r) * self.rho(r) * self.planet.DeltaS

    def LH(self, r):
        LH, i = integrate.quad(self.PL, 0, r)
        return LH

    def Pc(self, r):
        '''Secular cooling power (Eq. A8 Labrosse 2015)'''
        return -4. * np.pi / 3. * self.planet.rho_c * self.planet.CP * self.planet.L_rho**3. *\
                (1 - r**2. / self.planet.L_rho**2 - self.planet.A_rho* r**4. / self.planet.L_rho**4.)**(-self.planet.gamma) \
                * (self.dTL_dr_IC(r) + 2. * self.planet.gamma \
                * self.T_melt(r) * r / self.planet.L_rho**2. *(1 + 2. * self.planet.A_rho * r**2. / self.planet.L_rho**2.) \
                /(1 - r**2. / self.planet.L_rho**2. - self.planet.A_rho * r**4. / self.planet.L_rho**4.)) \
                * (self.fC(self.planet.r_OC / self.planet.L_rho, self.planet.gamma))

    def SC(self, r):
        SC, i = integrate.quad(self.Pc, 0, r)  
        return SC

    def Px(self, r):
        ''' Gravitational heat power (Eq. A14 Labrosse 2015)'''
        return 8 * np.pi**2 * self.planet.chi0 * self.planet.GC * self.planet.rho_c**2 * self.planet.beta * r**2. \
        * self.planet.L_rho**2. / self.fC(self.planet.r_OC / self.planet.L_rho, 0) \
        * (self.fX(self.planet.r_OC / self.planet.L_rho, r) - self.fX(r / self.planet.L_rho, r))

    def GH(self, r):
        GH, i = integrate.quad(self.Px, 0, r)
        return GH


    def energies(self):
        ''' Latent heat '''
        self.L = 4. * np.pi / 3. * self.planet.rho_c * self.planet.TL0 * self.planet.DeltaS * self.planet.r_IC**3. * (1 - 3. / 5. \
            * (1 + self.planet.K_c / self.planet.TL0 * self.planet.dTL_dP) * self.planet.r_IC**2. / self.planet.L_rho**2. \
            + self.planet.chi0 / (2 * self.fC(self.planet.r_OC / self.planet.L_rho, 0.) * self.planet.TL0) * self.planet.dTL_dchi * self.planet.r_IC**3. / self.planet.L_rho**3.)
        print("Latent heat", self.L,"J")
        
        ''' Secular cooling '''
        self.C = 4. * np.pi / 3. * self.planet.rho_c * self.planet.CP * self.planet.L_rho * self.planet.r_IC**2 * self.fC(self.planet.r_OC / self.planet.L_rho, self.planet.gamma)\
                * (self.planet.dTL_dP * self.planet.K_c - self.planet.gamma * self.planet.TL0 - self.planet.dTL_dchi * self.planet.chi0 / self.fC(self.planet.r_OC / self.planet.L_rho, 0.) * self.planet.r_IC / self.planet.L_rho)    
        print("Secular cooling", self.C,"J")
        self.C, i = integrate.quad(self.Pc, 0, self.planet.r_IC)
        # Why are you calculating C twice??
        print("Secular cooling", self.C,"J")

        ''' Gravitational energy '''
        self.G = 8 * np.pi**2. / 15. * self.planet.chi0 * self.planet.GC * self.planet.rho_c**2. * self.planet.beta * self.planet.r_IC**3. * self.planet.r_OC**5. / self.planet.L_rho**3. \
            / self.fC(self.planet.r_OC/self.planet.L_rho,0)*(1. - self.planet.r_IC**2 / self.planet.r_OC**2 + 3. * self.planet.r_IC**2. / 5. / self.planet.L_rho**2. \
                - 13. * self.planet.r_OC**2. / 14. / self.planet.L_rho**2. + 5./18. * self.planet.r_IC**3. * self.planet.L_rho**2. /self.planet.r_OC**5.)
        print("Gravitational energy", self.G,"J")

        ''' Total energy '''
        self.E_tot = self.L + self.C + self.G
        print("Total energy", self.E_tot,"J")


    def profiles(self, N=50):

        r_IC_vec = np.linspace(0, self.planet.r_IC, N)
        Q_cmb = np.linspace(3, 10, 4)*1e12   # From Labrosse+2001

        L_H = np.zeros(N)
        S_C = np.zeros(N)
        G_H = np.zeros(N)
        time = np.zeros(N)

        for i, radius_ic in enumerate(r_IC_vec):
            L_H[i] = self.LH(radius_ic)
            S_C[i] = self.SC(radius_ic)
            G_H[i] = self.GH(radius_ic)

        E_tot = L_H + S_C + G_H

        time = E_tot / self.planet.Q_CMB/(np.pi*1e7*1e9) # Q_CMB is a specific value!

        plt.figure(1)
        label = ['Latent heating','Secular cooling','Gravitational heating','Total energy']
        plt.plot(time, L_H,label=label[0])
        plt.plot(time, S_C,label=label[1]) 
        plt.plot(time, G_H, label=label[2])
        plt.plot(time, E_tot,label=label[3])
        plt.legend()
        plt.xlabel('Time (Gyr)')
        plt.ylabel('Energy (J)')

        plt.figure(2)
        plt.plot(time, r_IC_vec*1e-3,label='$Q_{\mathrm{CMB}}$=7.4e12 W')
        plt.xlabel('Time (Gyr)')
        plt.ylabel('Inner core size (km)')
        plt.legend()


        L_H= np.zeros((N,4),dtype=np.float32)
        S_C = np.zeros((N,4),dtype=np.float32)
        G_H = np.zeros((N,4),dtype=np.float32)
        time = np.zeros((N,4),dtype=np.float32)

        for i in range(len(r_IC_vec)):
            L_H[i] = self.LH(r_IC_vec[i])
            S_C[i] = self.SC(r_IC_vec[i])
            G_H[i] = self.GH(r_IC_vec[i])

        E_tot = L_H + S_C + G_H

        time = (L_H + S_C + G_H) / Q_cmb/(np.pi*1e7*1e9)

        plt.figure(3)
        plt.plot(time, r_IC_vec*1e-3)
        plt.xlabel('Time (Gyr)')
        plt.ylabel('Inner core size (km)')
        plt.show()




class Earth(Rocky_Planet):

    def parameters(self):

        self.read_parameters("Earth.yaml")
        # -----------------------------------------------------------------------------
        # Density length scale for Earth
        self.L_rho = np.sqrt(3. * self.K_c / (2. * np.pi * self.GC * self.rho_c**2.))
        # A_rho for Earth
        self.A_rho = self.A_rho()



Earth().evolution()
# Earth().read_parameters("Earth.yaml")

