import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import scipy.integrate as integrate
import yaml
import pandas as pd
import sys

year = 365.25*3600*24 #s
G = 6.67e-11

Mp = 1.2
XFe = 50
FeM = 0.00

qcmb_ev = pd.read_csv("qc_T_M{:02d}_Fe{:02d}_FeM{:02d}.txt".format(int(10*Mp),int(XFe)+5, int(FeM)), sep=" ", skiprows=1, header=None)
qcmb_ev.columns = ["time", "qcmb", "Tcmb"]   

# Define class for test case
class Rocky_Planet():
    """ Rocky planet, defines physical parameters and evolution """

    def __init__(self):
        self.parameters()

    def parameters(self):
        pass 

    def A_rho(self):
        return (5. * self.Kprime_0 - 13.) / 10.

    def read_parameters(self, file):
        with open(file, 'r') as stream:
            try:
                dict_param = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        for k, v in dict_param.items():
            setattr(self, k, float(v))

# Run evolution model          
    def evolution(self):
        evolution = Evolution(self)
        #for i in enumerate(time_vector):
#            if i==0:
#                self.r_IC = self.planet.r_IC_0
            #if self.planet.r_IC_0 == 0.0 or self.T0 < self.planet.TL0:
        evolution.noic()
            #else:
        evolution.ic()
        evolution.profiles()

class Evolution():
    """ Calculate the thermal evolution a given planet """

    def __init__(self, planet):
        self.planet = planet
        
    def dTL_dr_IC(self, r):
        ''' Melting temperature jump at ICB (to be modified) '''
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


    def PC(self, r):
        '''Secular cooling power (Eq. A8 Labrosse 2015)'''
        return -4. * np.pi / 3. * self.planet.rho_c * self.planet.CP * self.planet.L_rho**3. *\
                (1 - r**2. / self.planet.L_rho**2 - self.planet.A_rho* r**4. / self.planet.L_rho**4.)**(-self.planet.gamma) \
                * (self.dTL_dr_IC(r) + 2. * self.planet.gamma \
                * self.T_melt(r) * r / self.planet.L_rho**2. *(1 + 2. * self.planet.A_rho * r**2. / self.planet.L_rho**2.) \
                /(1 - r**2. / self.planet.L_rho**2. - self.planet.A_rho * r**4. / self.planet.L_rho**4.)) \
                * (self.fC(self.planet.r_OC / self.planet.L_rho, self.planet.gamma))


    def PX(self, r):
        ''' Gravitational heat power (Eq. A14 Labrosse 2015)'''
        return 8 * np.pi**2 * self.planet.chi0 * self.planet.GC * self.planet.rho_c**2 * self.planet.beta * r**2. \
        * self.planet.L_rho**2. / self.fC(self.planet.r_OC / self.planet.L_rho, 0) \
        * (self.fX(self.planet.r_OC / self.planet.L_rho, r) - self.fX(r / self.planet.L_rho, r))

    
    def noic(self):
        
        self.fC = self.fC(self.planet.r_OC / self.planet.L_rho, self.planet.gamma)
        
        ''' Secular cooling power '''
        self.PC = (-4*np.pi/3*self.planet.rho_c*self.planet.CP*self.planet.L_rho**3*self.fC)
        
        '''Temperature increase at center'''
        self.dT0_dt = self.planet.Q_CMB/self.PC  # leave QCMB constant for now
        
        ''' New central temperature '''
        self.T0 = self.planet.T0 + self.dT0_dt
        
        ''' Inner core growth '''
        self.dRic_dt = 0
        
        self.r_IC = 0
        
        ''' Latent heat power '''
        self.PL = 0

        ''' Gravitational heat power '''
        self.PX = 0
        
    def ic(self):
        
        self.PC = -4. * np.pi / 3. * self.planet.rho_c * self.planet.CP * self.planet.L_rho**3. *\
                (1 - self.r_IC**2. / self.planet.L_rho**2 - self.planet.A_rho* self.r_IC**4. / self.planet.L_rho**4.)**(-self.planet.gamma) \
                * (self.dTL_dr_IC(self.r_IC) + 2. * self.planet.gamma \
                * self.T_melt(self.r_IC) * self.r_IC / self.planet.L_rho**2. *(1 + 2. * self.planet.A_rho * self.r_IC**2. / self.planet.L_rho**2.) \
                /(1 - self.r_IC**2. / self.planet.L_rho**2. - self.planet.A_rho * self.r_IC**4. / self.planet.L_rho**4.)) \
                * (self.fC(self.planet.r_OC / self.planet.L_rho, self.planet.gamma))
                
        self.PL = 4. * np.pi * self.r_IC**2. * self.T_melt(self.r_IC) * self.rho(self.r_IC) * self.planet.DeltaS
        
        self.PX = 8 * np.pi**2 * self.planet.chi0 * self.planet.GC * self.planet.rho_c**2 * self.planet.beta * self.r_IC**2. \
                * self.planet.L_rho**2. / self.fC(self.planet.r_OC / self.planet.L_rho, 0) \
                * (self.fX(self.planet.r_OC / self.planet.L_rho, self.r_IC) - self.fX(self.r_IC / self.planet.L_rho, self.r_IC))
        
        self.dRic_dt = self.planet.Q_CMB/(self.PC + self.PL + self.PX)
        
        self.r_IC = self.r_IC + self.dRic_dt
        
#        ## ENERGIES
#        ''' Latent heat '''
#        self.L = 4. * np.pi / 3. * self.planet.rho_c * self.planet.TL0 * self.planet.DeltaS * self.planet.r_IC**3. * (1 - 3. / 5. \
#            * (1 + self.planet.K_c / self.planet.TL0 * self.planet.dTL_dP) * self.planet.r_IC**2. / self.planet.L_rho**2. \
#            + self.planet.chi0 / (2 * self.fC(self.planet.r_OC / self.planet.L_rho, 0.) * self.planet.TL0) * self.planet.dTL_dchi * self.planet.r_IC**3. / self.planet.L_rho**3.)
#        print("Latent heat", self.L,"J")
#        
#        ''' Secular cooling '''
#        self.C = 4. * np.pi / 3. * self.planet.rho_c * self.planet.CP * self.planet.L_rho * self.planet.r_IC**2 * self.fC(self.planet.r_OC / self.planet.L_rho, self.planet.gamma)\
#                * (self.planet.dTL_dP * self.planet.K_c - self.planet.gamma * self.planet.TL0 - self.planet.dTL_dchi * self.planet.chi0 / self.fC(self.planet.r_OC / self.planet.L_rho, 0.) * self.planet.r_IC / self.planet.L_rho)    
#        print("Secular cooling", self.C,"J")
#
#
#        ''' Gravitational energy '''
#        self.G = 8 * np.pi**2. / 15. * self.planet.chi0 * self.planet.GC * self.planet.rho_c**2. * self.planet.beta * self.planet.r_IC**3. * self.planet.r_OC**5. / self.planet.L_rho**3. \
#            / self.fC(self.planet.r_OC/self.planet.L_rho,0)*(1. - self.planet.r_IC**2 / self.planet.r_OC**2 + 3. * self.planet.r_IC**2. / 5. / self.planet.L_rho**2. \
#                - 13. * self.planet.r_OC**2. / 14. / self.planet.L_rho**2. + 5./18. * self.planet.r_IC**3. * self.planet.L_rho**2. /self.planet.r_OC**5.)
#        print("Gravitational energy", self.G,"J")
#
#        ''' Total energy '''
#        self.E_tot = self.L + self.C + self.G
#        print("Total energy", self.E_tot,"J")


    def profiles(self):
        
        time_vector =  qcmb_ev[qcmb_ev.columns[0]] # Time vector (years)
        Q_cmb = pd.Series([self.planet.Q_CMB]*len(time_vector)) # QCMB constant with time for now
        
        PC = np.zeros(len(time_vector))
        PL = np.zeros(len(time_vector))
        PX = np.zeros(len(time_vector))
        fC = np.zeros(len(time_vector))
        T0 = np.zeros(len(time_vector))
        dRic_dt = np.zeros(len(time_vector))
        r_IC = np.zeros(len(time_vector))

        r_IC_0 = self.planet.r_IC_0
        T0_0 = self.planet.T0

        for i in range(len(time_vector)):
            
            if r_IC_0 == 0 or T0[i]>self.planet.TL0:
                
                fC[i] = fC(self.planet.r_OC / self.planet.L_rho, self.planet.gamma)
                
                PC[i] = (-4*np.pi/3*self.planet.rho_0*self.planet.CP*self.planet.L_rho**3*fC[i])
                
                dT0_dt = self.planet.Q_CMB/PC[i]
                
                T0[i] = T0_0 + dT0_dt
        
                dRic_dt[i] = 0
                
                r_IC[i] = 0
        
                PL[i] = 0

                PX[i] = 0
                           
            else:
            
                PC[i] = self.PC(r_IC[i])
                
                PL[i] = self.PL(r_IC[i])
                
                PX[i] = self.PX(r_IC[i])
                
                dRic_dt[i] = Q_cmb[i]/(PC[i] + PL[i] + PX[i])
                
                # Update inner core radius
                r_IC = r_IC + dRic_dt[i]
                
        plt.figure(1)
        plt.plot(time_vector,r_IC)
        plt.xlabel('Time (yrs)')
        plt.ylabel('Inner core radius')
        plt.show()

# --------------------------------------------------------------------------- #

Mp = 1.2
XFe = 50
FeM = 0.00

class Exo(Rocky_Planet):
    
    #GC = 6.67384e-11 #m3/kg/s2

    def parameters(self):
        self.read_parameters("M_ {:.1f}_Fe_{:.0f}.0000_FeM_{:2.0f}.0000.yaml".format(Mp, XFe, FeM))
        qcmb_ev = pd.read_csv("qc_T_M{:02d}_Fe{:02d}_FeM{:02d}.txt".format(int(10*Mp),int(XFe)+5, int(FeM)), sep=" ", skiprows=1, header=None)
        qcmb_ev.columns = ["time", "qcmb", "Tcmb"]
        
Exo().evolution()        
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

