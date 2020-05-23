#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:05:11 2020

@author: irenebonati
"""

import numpy as np
from scipy.optimize import minimize_scalar
import sympy as sp


Labrosse = "True"

GC = 6.67430e-11

Mp = 1.
XFe = 35.
FeM = 0.
S = 0.

#A_rho = 0.4877119737341147
#CP = 1180.5824183453947
#DeltaS =  127.0
#Deltarho_ICB =  500.0
#K_c =  1403000000000.0
#L_rho = 7731750.401722977
#P0 = 409.7364106
#Pcmb = 137.54334939999998
#T0 = 7261.7707097
#TL0= 7155.525966138659
#Tcmb = 5343.7402443
#alpha_c = 1.6958213383412207e-05
#beta = 0.83
#chi0 = 0.056
#dTL_dP = 9.0e-09
#dTL_dchi = -21000.0
#gamma = 1.266429415122085
#gc = 8.294367392473594
#k_c = 163.0
#r_IC_0 = 0.0
#r_OC = 3437735.3620821
#r_planet = 6368890.629103
#rho_0 = 13980.295428854699

A_rho = 0.484
CP = 750.
DeltaS =  127.0
Deltarho_ICB =  500.0
K_c =  1403000000000.0
L_rho = 8039e3
P0 = 409.7364106
T0 = 5985.
TL0= 5700.
Tcmb = 5343.7402443
alpha_c = 1.6958213383412207e-05
beta = 0.83
chi0 = 0.056
dTL_dP = 9.0e-09
dTL_dchi = -21000.0
gamma = 1.5
k_c = 163.0
r_IC_0 = 0.0
r_OC = 3480e3
r_planet = 6371000.
rho_0 = 12502.

'''Find inner core radius when it first starts forming'''
def find_r_IC(T, S):
    def Delta_T(r):
        P = pressure_diff(r)+P0
        Ta = T_adiabat(r,T)
        TL = T_liquidus_core(r)
        return (Ta - TL)**2
    res = minimize_scalar(Delta_T, bounds=(0., 6e6), method='bounded')
    if not res.success:
        print("find_r_IC didn't converge")
    r_IC = res.x
    if r_IC < 1: r_IC = np.array(0.)
    return r_IC.tolist()

def pressure_diff(r):  
    '''Pressure difference (GPa)'''
    GC = 6.67430e-11
    K0 = L_rho**2/3.*2.*np.pi*GC*rho_0**2 /1e9 #in GPa
    K0 = (2./3. * np.pi * L_rho**2 * rho_0**2 *GC)/1e9
    factor = (r**2/L_rho**2)-(4.*r**4)/(5.*L_rho**4)
    return -K0*factor

def T_adiabat(r,T): # T is the central value as a function of time!!! Is the inner core isothermal?
    '''Adiabatic temperature''' 
    return T*(1-r**2/L_rho**2-A_rho*r**4/L_rho**4)**gamma

def _PL(r):
    '''Latent heat power'''
    return 4. * np.pi * r**2. * T_liquidus_core(r) * rho(r) * DeltaS # Is rho correct? Yes!

def _PC(r):
    '''Secular cooling power (Eq. A8 Labrosse 2015)'''
    return -4. * np.pi / 3. * rho_0 * CP * L_rho**3. *(1 - r**2. / L_rho**2 - A_rho* r**4. / L_rho**4.)**(-gamma) * (dTL_dr_IC(r) + 2. * gamma * T_liquidus_core(r) * r / L_rho**2. *(1 + 2. * A_rho * r**2. / L_rho**2.) /(1 - r**2. / L_rho**2. - A_rho * r**4. / L_rho**4.)) * (_fC(r_OC / L_rho, gamma)-_fC(r / L_rho, gamma))

def _PX(r):
    ''' Gravitational heat power (Eq. A14 Labrosse 2015)'''
    return 8 * np.pi**2 * chi0 * GC * rho_0**2 * beta * r**2. * L_rho**2. / _fC(r_OC / L_rho, 0) * (fX(r_OC / L_rho, r) - fX(r / L_rho, r))
                   

def M_OC(r):
    '''Equation M_OC(t) in our paper'''
    return 4./3. * np.pi * rho_0 * L_rho**3 * (_fC(r_OC/L_rho,0)-_fC(r/L_rho,0))
    
    
def _fC(r, delta): 
    '''fC (Eq. A1 Labrosse 2015)'''
    return r**3. * (1 - 3. / 5. * (delta + 1) * r**2.- 3. / 14. * (delta + 1) * (2 * A_rho - delta) * r**4.)

def fX(x, r):
    '''fX (Eq. A15 Labrosse 2015)'''
    return (x)**3. * (-r**2. / 3. / L_rho**2. + 1./5. * (1.+(r**2)/L_rho**2.) *(x)**2.-13./70. * (x)**4.) 


def rho(r):
    ''' Density (Eq. 5 Labrosse 2015)'''
    return rho_0 * (1. - r**2. / L_rho**2. - A_rho * r**4. / L_rho**4.)

   
def T_liquidus_core(r):
    ''' Melting temperature (Eq. 14 Labrosse 2015)'''
    return TL0 - K_c * dTL_dP * r**2. / L_rho**2. + dTL_dchi * chi0 * r**3./ (L_rho**3. * _fC(r_OC / L_rho, 0.))
            
def dTL_dr_IC(r):
    result = -K_c * 2.*dTL_dP * r / L_rho**2. + 3. * dTL_dchi * chi0 * r**2. / (L_rho**3. * _fC(r_OC / L_rho, 0.))
    return result
    

def test(T, dt): 
    
    r_IC = find_r_IC(T,S)
    print ("r_IC = ", r_IC)
    P_IC = pressure_diff(r_IC)+P0     
    print ("P_IC = ",P_IC)   

    print("Testing at ", T, " and ", r_IC)
    '''Secular cooling power'''
    PC = _PC(r_IC)
    
    '''Latent heat power'''
    PL = _PL(r_IC)
    
    '''Gravitational heat power'''
    PX =_PX(r_IC)
    
    QC = 7e12
    drIC_dt = QC/(PC + PL + PX)
    r_IC += drIC_dt * dt
    P_IC = pressure_diff(r_IC) + P0
    
    '''Temperature at the ICB'''        
    Tlatent = T_liquidus_core(r_IC)
    
    fC = _fC(r_OC / L_rho, gamma)
            
    ''' Secular cooling power '''
    PC = (-4*np.pi/3*rho_0*CP*L_rho**3*fC)                
 
    '''CMB heat flow'''
    Q_CMB = 7e12 #4*np.pi*self.planet.r_OC**2*qcmb

    '''Temperature change at center'''
    dT_dt = Q_CMB/PC 
    
    ''' New central temperature '''
    Tnolatent = T + dT_dt * dt 
    
    print ("Tlatent = ",Tlatent, "should be higher than Tnolatent = ",Tnolatent)
    return (Tlatent , Tnolatent)

#Tl, Tn = test(5680,1e5)
#assert Tl > Tn, (Tl, Tn)

r_IC_nolatent = find_r_IC(5680, 0)
print ("r_IC_nolatent = ", r_IC_nolatent/1e3)

r_IC_latent = find_r_IC(5690, 0)
print ("r_IC_latent = ", r_IC_latent/1e3)

# T latent should be higher than T_nolatent!!!
# r_IC latent should be smaller than r_IC_nolatent!!

