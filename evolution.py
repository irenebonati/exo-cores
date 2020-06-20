#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:51:24 2020

@author: irenebonati
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
#import sympy as sp
#from sympy import lambdify
from scipy.optimize import minimize_scalar

year              = 365.25*3600*24    # 1 year (s)
GC                = 6.67430e-11
Ak                = 2.39              # Radial dependence of conductivity
beta              = 0.2               # Saturation constant for fast rotating polar dynamos
mu_0              = 4*np.pi*1e-7      # Magnetic permeability (Hm-1)
M_Earth           = 5.972e24          # Mass of the Earth (kg)
k_c               = 150               # Core conductivity (estimated)
magn_moment_Earth = 7.8e22            # Magnetic moment Earth (Am2)

class Evolution():

    def __init__(self, planet):
        self.planet = planet
        
        '''Initialize matrices'''
        
        # self.idx = {
        #   'PC':4,
        #   'PL':5,
        # }
        # self.PC <=> self.giant[:,self.idx['PC']]
        # PC = self.giant...
        
        self.r_IC = np.zeros_like(self.planet.time_vector)
        self.drIC_dt = np.zeros_like(self.planet.time_vector)
        self.T = np.zeros_like(self.planet.time_vector)
        self.dT_dt = np.zeros_like(self.planet.time_vector)
        self.PC = np.zeros_like(self.planet.time_vector)
        self.PL = np.zeros_like(self.planet.time_vector)
        self.PX = np.zeros_like(self.planet.time_vector)
        self.QC = np.zeros_like(self.planet.time_vector)
        self.QL = np.zeros_like(self.planet.time_vector)
        self.QX = np.zeros_like(self.planet.time_vector)
        self.Q_CMB = np.zeros_like(self.planet.time_vector)
        self.T_CMB = np.zeros_like(self.planet.time_vector)
        self.SC = np.zeros_like(self.planet.time_vector)
        self.TC = np.zeros_like(self.planet.time_vector)
        self.SR = np.zeros_like(self.planet.time_vector)
        self.TR = np.zeros_like(self.planet.time_vector)
        self.Sk = np.zeros_like(self.planet.time_vector)
        self.Tphi = np.zeros_like(self.planet.time_vector)
        self.phi = np.zeros_like(self.planet.time_vector)
        self.qc_ad = np.zeros_like(self.planet.time_vector)
        self.F_th = np.zeros_like(self.planet.time_vector) 
        self.F_X = np.zeros_like(self.planet.time_vector)
        self.Bc = np.zeros_like(self.planet.time_vector)
        self.Bs = np.zeros_like(self.planet.time_vector)
        self.M = np.zeros_like(self.planet.time_vector)
        self.M_ratio = np.zeros_like(self.planet.time_vector)
        self.P_IC = np.zeros_like(self.planet.time_vector)
        self.S_t = np.zeros_like(self.planet.time_vector)
        self.Delta_time = self.planet.time_vector.diff()*year
        self.t_IC0 = 0
        self.T_IC0 = 0
        
# ------------------------------------------------------------------------------------------------------------------- #
            
        '''Find initial inner core size and melting temperature'''
        TL0 = self.T_liquidus_core(self.planet.P0, self.planet.S)
        r_IC_0 = self.find_r_IC(self.planet.T0,self.planet.S)
        
        '''Check if the radius and melting temperature I find correspond to the ones in the yaml file'''
        if self.planet.S == 0:
            assert int(TL0) == int(self.planet.TL0_0),int(r_IC_0) == int(self.planet.r_IC_0)
        elif self.planet.S == 0.05:
            assert int(TL0) == int(self.planet.TL0_005),int(r_IC_0) == int(self.planet.r_IC_005)
        elif self.planet.S == 0.11:
            assert int(TL0) == int(self.planet.TL0_011),int(r_IC_0) == int(self.planet.r_IC_011)
        
        '''Rename variables for later calculations in the code'''
        self.planet.TL0 = TL0
        self.planet.r_IC_0 = r_IC_0
                        
        if self.planet.r_IC_0 == 0.0:
            '''If no initial inner core, P0 and T0 are same as in yaml parameter file'''
            self.T[0] = self.planet.T0 
            self.P_IC[0] = self.planet.P0
            self.S_t[0] = self.planet.S

        else:                  
            '''If initial inner core, define T by using the melting temperature by Stixrude'''
            '''Define P by calculating the pressure at the ICB radius'''
            self.P_IC[0] = self.pressure_diff(self.planet.r_IC_0)+self.planet.P0
            self.S_t[0] = self.planet.S * self.M_OC(0)/self.M_OC(self.planet.r_IC_0)
            self.T[0] = self.T_liquidus_core(self.P_IC[0], self.S_t[0])            
            #self.T[0] = self.T_melt(self.planet.r_IC_0)
            
        '''Initial inner core radius, CMB heat flux set to 0'''
        '''I do this because the first value in the file of Lena is negative'''
        self.r_IC[0] = self.planet.r_IC_0
        self.T_CMB[0] = self.T_adiabat(self.planet.r_OC,self.T[0])

# ------------------------------------------------------------------------------------------------------------------- #
        
    '''Run evolution model'''         
    def run(self):
        
        for i,time in enumerate(self.planet.time_vector[1:]):
            
            '''No initial inner core --> update_noic routine'''
            if self.r_IC[i] == 0.0 and self.T[i] > self.planet.TL0:
                              
                T, dT_dt,r_IC, drIC_dt, PC, PL, PX, Q_CMB, T_CMB, QC, QL, QX,qc_ad, F_th, F_X, Bc, Bs, M,M_ratio,P_IC,S_t =  self.update_noic(self.T[i],self.Delta_time[i+1],self.planet.qcmb[i])
                '''Shift updated value to the next time step'''
                T,r_IC, drIC_dt, PC, PL, PX, Q_CMB, T_CMB, QC, QL, QX,qc_ad, F_th, F_X, Bc, Bs, M, M_ratio, P_IC,S_t = self.update_value(T,r_IC, drIC_dt, PC, PL, PX, Q_CMB, T_CMB, QC, QL, QX,qc_ad, F_th, F_X, Bc, Bs, M, M_ratio, P_IC,S_t,i)               
                assert abs(Q_CMB-self.planet.qcmb[i]*self.planet.r_OC**2 * 4 * np.pi) < 1., (Q_CMB/1e13,(self.planet.qcmb[i]*self.planet.r_OC**2 * 4 * np.pi)/1e13)
               
                '''If T is lower than Tmelt we start forming an inner core'''
                if self.T[i+1] < self.planet.TL0:
                    
                    # At what time does an inner core start to form?
                    self.t_IC0 = self.planet.time_vector[i]
                    self.T_IC0 = self.T[i]

                    # IC radius and ICB pressure at new time step with T>Tmelt
                    r_IC_form = self.find_r_IC(self.T[i+1],self.planet.S)
                    P_IC_form = self.pressure_diff(r_IC_form)+self.planet.P0
                    T_form = self.T[i+1]
                    
                    
                    Tmelt = self.planet.TL0
                    ratio = (self.T[i]-Tmelt)/(self.T[i]-self.T[i+1])
                    #print ("ratio_NOIC = ",ratio)
                    assert 0 < ratio < 1, ratio

                    Delta_t_IC = ratio * self.Delta_time[i+1] # Time step until Tmelt is reached
           
                    dt_rem = self.Delta_time[i+1]-Delta_t_IC  # Remaining time step
                    
                    ''' Go "back" and calculate the heat budget until an inner core starts forming --> use Delta_t_IC'''
                    T, dT_dt,r_IC, drIC_dt, PC, PL, PX, Q_CMB, T_CMB, QC, QL, QX,qc_ad, F_th, F_X, Bc, Bs, M,M_ratio,P_IC,S_t =  self.update_noic(self.T[i],Delta_t_IC,self.planet.qcmb[i])
                    '''Shift updated value to the next time step'''
                    T,r_IC, drIC_dt, PC, PL, PX, Q_CMB, T_CMB, QC, QL, QX,qc_ad, F_th, F_X, Bc, Bs, M, M_ratio, P_IC,S_t = self.update_value(T,r_IC, drIC_dt, PC, PL, PX, Q_CMB, T_CMB, QC, QL, QX,qc_ad, F_th, F_X, Bc, Bs, M, M_ratio, P_IC,S_t,i)                    
                    
                    '''Take a small initial inner core radius in order not to overshoot heat budget terms'''
                    r_IC_0 = 1e3
                    P_IC_0 = self.pressure_diff(r_IC_0)+self.planet.P0         

                    '''Slowly start growing an inner core'''
                    timesteps = 100
                    tmp=0
                    sum_ratio=0
                    Q_CMB_0= 0
                    for m in range(timesteps):
                        dt = dt_rem/timesteps #self.Delta_time[i+1]/timesteps#dt_rem/2**(timesteps-m-1) 
                        tmp += dt
                            
                        ratio_0 = dt/self.Delta_time[i+1]
                        sum_ratio+=ratio_0

                        '''With inner core --> update_ic routine and use dt'''  
                        T, r_IC, drIC_dt, PC, PL, PX, Q_CMB ,T_CMB, QC, QL, QX,qc_ad, F_th, F_X, Bc, Bs, M,M_ratio,P_IC,S_t =  self.update_ic(r_IC_0, dt,self.planet.qcmb[i],P_IC_0,self.planet.S,ratio=ratio_0)

                        r_IC_0 = r_IC
                        P_IC_0 = P_IC
                        Q_CMB_0 += Q_CMB  
                        
                        if m ==timesteps-1:
                            Q_CMB = (sum_ratio+ratio)*4*np.pi*self.planet.qcmb[i]*self.planet.r_OC**2
                            drIC_dt = Q_CMB/(PC + PL + PX)
                            QL = PL*drIC_dt
                            QC = PC*drIC_dt
                            QX = PX*drIC_dt

                        T,r_IC, drIC_dt, PC, PL, PX, Q_CMB, T_CMB, QC, QL, QX,qc_ad, F_th, F_X, Bc, Bs, M, M_ratio, P_IC,S_t = self.update_value(T,r_IC, drIC_dt, PC, PL, PX, Q_CMB, T_CMB, QC, QL, QX,qc_ad, F_th, F_X, Bc, Bs, M, M_ratio, P_IC,S_t,i)
                    #print((1-ratio)*self.planet.qcmb[i]*self.planet.r_OC**2 * 4 * np.pi/1e13)
                    #assert tmp == self.Delta_time[i+1]-Delta_t_IC, (tmp, self.Delta_time[i+1]-Delta_t_IC)
                    #assert dt_rem == 0, self.Delta_time[i+1]
                    #assert r_IC < r_IC_form, (r_IC,r_IC_form)
                    #assert T > T_form, (T, T_form)
                    #print (T,T_form)
                    assert abs(Q_CMB-self.planet.qcmb[i]*self.planet.r_OC**2 * 4 * np.pi) < 1., (Q_CMB/1e13,(self.planet.qcmb[i]*self.planet.r_OC**2 * 4 * np.pi)/1e13)
                    assert abs(QC + QL + QX - Q_CMB)<1,(QC + QL + QX,Q_CMB)
            else: 

                    dt_rem = self.Delta_time[i+1]
                    if self.r_IC[i] < 1e5:
                        timesteps = 200
                    elif self.r_IC[i] < 2e5:
                        timesteps = 50
                    else:
                        timesteps = 20
                    sum_ratio = 0
                    Q_CMB_0 = 0
                    r_IC = self.r_IC[i]
                    for m in range(timesteps):

                        dt = self.Delta_time[i+1]/timesteps
                        ratio_0 = dt/self.Delta_time[i+1]
                        sum_ratio+=ratio_0
                        dt_rem-=dt
                        
                        '''Initial inner core --> update_ic routine'''  
                        T, r_IC, drIC_dt, PC, PL, PX, Q_CMB, T_CMB, QC, QL, QX,qc_ad, F_th, F_X, Bc, Bs, M,M_ratio,P_IC,S_t =  self.update_ic(r_IC, dt,self.planet.qcmb[i],self.P_IC[i],self.S_t[i],ratio=ratio_0)
                        Q_CMB_0 += Q_CMB
                        T,r_IC, drIC_dt, PC, PL, PX, Q_CMB, T_CMB, QC, QL, QX,qc_ad, F_th, F_X, Bc, Bs, M, M_ratio, P_IC,S_t = self.update_value(T,r_IC, drIC_dt, PC, PL, PX, Q_CMB, T_CMB, QC, QL, QX,qc_ad, F_th, F_X, Bc, Bs, M, M_ratio, P_IC,S_t,i)
                    #assert dt_rem == 0, self.Delta_time[i+1]
                    assert abs(Q_CMB-(sum_ratio)*self.planet.qcmb[i]*self.planet.r_OC**2 * 4 * np.pi) < 1., (Q_CMB/1e13,(self.planet.qcmb[i]*self.planet.r_OC**2 * 4 * np.pi)/1e13)
# ------------------------------------------------------------------------------------------------------------------- #
         
        self.t_mf = 0 # 5 billion years
        '''Magnetic field lifetime routine'''
        for i in range(1,len(self.planet.time_vector)-1):
            if i==1 and self.M[i] !=0 and self.M[i+1]!=0:
                t_start = 0
            if i!=1 and self.M[i] !=0 and self.M[i+1]!=0 and self.M[i-1]==0:
                t_start=self.planet.time_vector[i]
            if self.M[i+1]==0 and self.M[i]!=0 and self.M[i-1]!=0:
                t_end = self.planet.time_vector[i+1]
                self.t_mf =(t_end-t_start)*1e-9
                break
            
        if self.M[1:].all()>0:
                self.t_mf = 5.
        if self.t_mf > 5.:
            self.t_mf = 5.
        #print ("The magnetic field lifetime is %.2f billion years."%(self.t_mf))     

# ------------------------------------------------------------------------------------------------------------------- #
    def plot(self,plots_folder):            
            '''Figures'''
            plt.figure()
#            for i in range(len(self.r_IC)-1):
#                if self.r_IC[i+1] > 0 and self.r_IC[i] == 0:
#                    t_IC = self.planet.time_vector[i]
#                    T_IC = self.T[i]
            ax1 = plt.gca()
            ax1.plot(self.planet.time_vector,self.T, color='rebeccapurple')
            ax1.set_ylabel('Temperature at the center/ICB (K)',color='rebeccapurple')
            ax1.set_xlabel('Time(years)')
            plt.gca().set_xlim([0,5e9]) 
            ax1.tick_params(axis='y', labelcolor='rebeccapurple')
            ax2 = ax1.twinx()  
            ax2.plot(self.planet.time_vector,self.r_IC/1e3, color='tomato')
            ax2.set_ylabel('Inner core radius (km)',color='tomato')
            ax2.tick_params(axis='y', labelcolor='tomato')
            #ax2.scatter(self.t_IC0,0,s=40,c='k',marker='*')
            #ax1.scatter(self.t_IC0,self.T_IC0,s=40,c='k',marker='*')
            plt.savefig(plots_folder + 'T+r_IC_{}ME_{}XFe_{}FeM.pdf'.format(self.planet.Mp,self.planet.XFe,self.planet.FeM), bbox_inches="tight")
            plt.show()
            
            if self.planet.S>0:
                plt.figure()
                plt.plot(self.planet.time_vector,self.S_t *100, color='crimson')
                plt.ylabel('Fraction of light elements (%)')
                plt.xlabel('Time(years)')
                plt.xlim([0,5e9])
                plt.ylim([0,110])
                plt.savefig(plots_folder + 'LE_{}ME_{}XFe_{}FeM.pdf'.format(self.planet.Mp,self.planet.XFe,self.planet.FeM), bbox_inches="tight")
                plt.show()
            
            plt.figure()
            plt.plot(self.planet.time_vector[1:],self.QC[1:], label='Secular cooling',color='lightseagreen')
            plt.plot(self.planet.time_vector[1:],self.QL[1:],label='Latent heat',color='sandybrown')
            plt.plot(self.planet.time_vector[1:],self.QX[1:], label='Gravitational heat',color='pink')
            plt.plot(self.planet.time_vector[1:],self.QL[1:]+self.QC[1:]+self.QX[1:], label='Total ($Q_{\mathrm{CMB}}$)',color='navy')
            plt.xlabel('Time (years)')
            plt.ylabel('Contributions to energy balance (W)')
            plt.title('$M=$ %.1f $M_{\oplus}$, $C_{\mathrm{Fe}}=$ %.0f wt %%' %(np.float(self.planet.Mp),np.float(self.planet.XFe)))
            plt.gca().set_xlim(left=self.planet.time_vector[1])
            plt.legend()
            plt.xlim([0,5e9])
            plt.savefig(plots_folder + 'Energy_balance_{}ME_{}XFe_{}FeM.pdf'.format(self.planet.Mp,self.planet.XFe,self.planet.FeM), bbox_inches="tight")
            plt.show()
            
            fig, ax = plt.subplots(1, 2, figsize=[10,4],sharex=True)
            ax[0].plot(self.planet.time_vector,self.Q_CMB,color='royalblue')
            ax[0].set_ylabel('CMB heat flow (W)')
            ax[0].set_xlabel('Time (years)')
            plt.xlim([0,5e9])  
            ax[1].plot(self.planet.time_vector,self.T_CMB,color='royalblue')
            ax[1].set_xlabel('Time (years)')
            ax[1].set_ylabel('CMB temperature (K)')
            plt.suptitle('$M=$ %.1f $M_{\oplus}$, $C_{\mathrm{Fe}}=$ %.0f wt %%' %(np.float(self.planet.Mp),np.float(self.planet.XFe)))
            plt.subplots_adjust(wspace=0.4)
            plt.savefig(plots_folder + 'QCMB_TCMB_{}ME_{}XFe_{}FeM.pdf'.format(self.planet.Mp,self.planet.XFe,self.planet.FeM), bbox_inches="tight")
            plt.show()
            
            plt.figure()
            plt.plot(self.planet.time_vector,self.F_th, label='Temperature',color='tomato')
            plt.plot(self.planet.time_vector,self.F_X, label='Composition',color='mediumseagreen')
            plt.xlabel('Time (years)')
            plt.ylabel('Buoyancy fluxes ($m^{2}s^{-3}$)')
            plt.title('$M=$ %.1f $M_{\oplus}$, $C_{\mathrm{Fe}}=$ %.0f wt %%' %(np.float(self.planet.Mp),np.float(self.planet.XFe)))
            plt.gca().set_xlim(left=self.planet.time_vector[1]) 
            plt.legend()
            plt.xlim([0,5e9])
            plt.savefig(plots_folder + 'Fluxes_{}ME_{}XFe_{}FeM.pdf'.format(self.planet.Mp,self.planet.XFe,self.planet.FeM), bbox_inches="tight")
            plt.show()
             
            fig, ax = plt.subplots(1, 2, figsize=[10,4],sharex=True)
            ax[0].plot(self.planet.time_vector,self.Bc * 1e3,label='CMB',color='tomato')
            ax[0].plot(self.planet.time_vector,self.Bs * 1e3,label='Surface',color='mediumseagreen')
            ax[0].set_xlabel('Time (years)')
            ax[0].set_ylabel('rms dipole field (mT)')
            ax[0].semilogy()
            plt.xlim([0,5e9]) 
            ax[0].legend()
            ax[1].plot(self.planet.time_vector,self.M,color='grey')
            ax[1].set_ylabel('Magnetic moment ($A m^{2}$)')
            ax2 = ax[1].twinx()  
            ax2.set_ylabel('Magnetic moment present Earth ($A m^{2}$)')  
            ax2.plot(self.planet.time_vector,self.M_ratio,color='grey')
            ax2.tick_params(axis='y')
            ax2.set_xlabel('Time (years)')
            plt.suptitle('$M=$ %.1f $M_{\oplus}$, $C_{\mathrm{Fe}}=$ %.0f wt %%' %(np.float(self.planet.Mp),np.float(self.planet.XFe)))
            plt.subplots_adjust(wspace=0.4)
            plt.savefig(plots_folder + 'MField_{}ME_{}XFe_{}FeM.pdf'.format(self.planet.Mp,self.planet.XFe,self.planet.FeM), bbox_inches="tight")
            plt.show()
            
# ------------------------------------------------------------------------------------------------------------------- #

    '''Routine for no initial inner core'''
    def update_noic(self,T,Delta_time,qcmb):
        
        fC = self.fC(self.planet.r_OC / self.planet.L_rho, self.planet.gamma)
                        
        ''' Secular cooling power '''
        PC = (-4*np.pi/3*self.planet.rho_0*self.planet.CP*self.planet.L_rho**3*fC)
                
        ''' Latent heat power '''
        PL = 0.

        ''' Gravitational heat power '''
        PX = 0.      
        
        '''CMB heat flow'''
        Q_CMB = 4*np.pi*self.planet.r_OC**2*qcmb
        assert Q_CMB > 0,Q_CMB

        '''Temperature change at center'''
        dT_dt = Q_CMB/PC 
        
        ''' New central temperature '''
        T = T + dT_dt * Delta_time 
        
        ''' Inner core growth '''
        drIC_dt = 0.
        
        ''' Inner core size'''
        r_IC = self.planet.r_IC_0      
        assert r_IC == 0
        
        P_IC = self.planet.P0
                
        '''Temperature at CMB'''
        T_CMB = self.T_adiabat(self.planet.r_OC,T)
        
        '''Secular cooling power'''
        QC = PC * dT_dt
        
        ''' Latent heat power '''
        QL = 0.

        ''' Gravitational heat power '''
        QX = 0. 
        
        rho_OC = self._density(self.planet.r_OC)
        
        '''Isentropic heat flux'''
        qc_ad = self._qc_ad(k_c,T_CMB,rho_OC)
        QC_ad = qc_ad *self.planet.r_OC**2*qcmb
        #assert QC_ad<Q_CMB
        
        '''Thermal buoyancy'''
        F_th = self._F_th(qcmb,qc_ad)
        
        '''Compositional buoyancy'''
        F_X = 0 
                
        S_t = self.planet.S
        
        '''rms dipole field @ CMB'''
        Bc = self._Bc(rho_OC,F_th,F_X,r_IC)
        
        '''rms dipole field @ surface'''
        Bs = self._Bs (Bc,self.planet.r_planet)
        
        '''Magnetic moment (Am2)'''
        M = self._magn_moment(rho_OC,F_th,F_X,r_IC)
        
        M_ratio = M/magn_moment_Earth
                
        return T, dT_dt,r_IC, drIC_dt, PC, PL, PX, Q_CMB, T_CMB, QC, QL, QX, qc_ad, F_th, F_X, Bc, Bs, M, M_ratio, P_IC,S_t
        
    '''Routine for initial inner core'''    
    def update_ic(self, r_IC, Delta_time,qcmb,P_IC,S_t,ratio=1):
        '''Secular cooling power'''
        PC = self._PC(r_IC,P_IC,S_t)
        
        '''Latent heat power'''
        PL = self._PL(r_IC,P_IC,S_t)
        
        '''Gravitational heat power'''
        PX = self._PX(r_IC)
        
        '''CMB heat flow'''
        Q_CMB = 4*np.pi*self.planet.r_OC**2*qcmb 
                        
        if r_IC > self.planet.r_OC or r_IC == self.planet.r_OC:
            r_IC = self.planet.r_OC
            drIC_dt = 0
            QC = Q_CMB
            QL = 0.
            QX = 0.
        else:    
            '''Inner core growth rate'''
            drIC_dt = Q_CMB/(PC + PL + PX)                         
            '''Inner core radius'''
            r_IC = r_IC + drIC_dt * Delta_time
            '''Secular cooling power'''
            QC = PC*drIC_dt
            '''Latent heat power'''
            QL = PL*drIC_dt       
            '''Gravitational heat power'''
            QX = PX*drIC_dt 
                                                                        
        P_IC = self.pressure_diff(r_IC) + self.planet.P0
                
        S_t = self._S_t(self.planet.S,r_IC)

        '''ICB temperature'''
        T = self.T_liquidus_core(P_IC, S_t)
                    
        '''CMB temperature'''
        T_CMB = self.T_adiabat(self.planet.r_OC,T)
        
        rho_OC = self._density(self.planet.r_OC)
                
        '''Isentropic heat flux'''
        qc_ad = self._qc_ad(k_c,T_CMB,rho_OC)
        QC_ad = qc_ad *self.planet.r_OC**2*qcmb
#        assert QC_ad<Q_CMB
 
        '''Thermal buoyancy'''
        F_th = self._F_th(qcmb,qc_ad)
        
        '''Compositional buoyancy'''
        F_X = self._F_X(r_IC,drIC_dt,S_t)
                                
        '''rms dipole field @ CMB'''
        Bc = self._Bc(rho_OC,F_th,F_X,r_IC)
        
        '''rms dipole field @ surface'''
        Bs = self._Bs (Bc,self.planet.r_planet)
        
        if r_IC > self.planet.r_OC or r_IC == self.planet.r_OC:
            M = 0.
            M_ratio = 0.
        else:
            '''Magnetic moment (Am2)'''
            M = self._magn_moment(rho_OC,F_th,F_X,r_IC)
             
            M_ratio = M/magn_moment_Earth
                                                        
        return T,r_IC, drIC_dt, PC, PL, PX, Q_CMB, T_CMB, QC, QL, QX,qc_ad, F_th, F_X, Bc, Bs, M, M_ratio, P_IC,S_t
    
    def update_value(self,T,r_IC, drIC_dt, PC, PL, PX, Q_CMB, T_CMB, QC, QL, QX,qc_ad, F_th, F_X, Bc, Bs, M, M_ratio, P_IC,S_t,i):
        self.T[i+1] = T
        self.r_IC[i+1] = r_IC
        self.drIC_dt[i+1] = drIC_dt
        self.PC[i+1] = PC
        self.PL[i+1] = PL
        self.PX[i+1] = PX
        self.Q_CMB[i+1] = Q_CMB
        self.T_CMB[i+1] = T_CMB   
        self.QC[i+1] = QC
        self.QL[i+1] = QL
        self.QX[i+1] = QX
        self.qc_ad[i+1] = qc_ad
        self.F_th[i+1] = F_th
        self.F_X[i+1] = F_X
        self.Bc[i+1] = Bc
        self.Bs[i+1] = Bs
        self.M[i+1] = M
        self.M_ratio[i+1] = M_ratio
        self.P_IC[i+1] = P_IC
        self.S_t[i+1] = S_t
        
        return self.T[i+1],self.r_IC[i+1], self.drIC_dt[i+1],self.PC[i+1],self.PL[i+1],self.PX[i+1],self.Q_CMB[i+1],self.T_CMB[i+1],self.QC[i+1],self.QL[i+1],self.QX[i+1],self.qc_ad[i+1],self.F_th[i+1],self.F_X[i+1],self.Bc[i+1],self.Bs[i+1],self.M[i+1],self.M_ratio[i+1],self.P_IC[i+1],self.S_t[i+1]
# ------------------------------------------------------------------------------------------------------------------- #   

    def dTL_dr_IC(self,x,S):
        
        K0 = (2./3. * np.pi * self.planet.L_rho**2 * self.planet.rho_0**2 *GC)/1e9
        L_rho = self.planet.L_rho
        P0 = self.planet.P0
        rho_0 = self.planet.rho_0
        r_OC =self.planet.r_OC
        M_OC_0 = self.M_OC(0)
        A_rho = self.planet.A_rho  

        der = -9.84558823529412*K0*(-0.00294117647058824*K0*(x**2/L_rho**2 - 0.8*x**4/L_rho**4) + 0.00294117647058824*P0)**(-0.485) \
           *(2*x/L_rho**2 - 3.2*x**3/L_rho**4)/(-np.log(1.0 - 0.75*M_OC_0*S/(np.pi*L_rho**3*rho_0*((r_OC/L_rho)**3.0*(-0.428571428571429 \
           *A_rho*(r_OC/L_rho)**4.0 - 0.6*(r_OC/L_rho)**2.0 + 1.0) - (x/L_rho)**3.0*(-0.428571428571429*A_rho*(x/L_rho)**4.0 - 0.6*(x/L_rho)**2.0 + 1.0)))) + 1.0) \
           - 4875.0*M_OC_0*S*(-0.00294117647058824*K0*(x**2/L_rho**2 - 0.8*x**4/L_rho**4) + 0.00294117647058824*P0)**0.515*((x/L_rho)**3.0*(-1.71428571428571*A_rho*(x/L_rho)**4.0 \
           /x - 1.2*(x/L_rho)**2.0/x) + 3.0*(x/L_rho)**3.0*(-0.428571428571429*A_rho*(x/L_rho)**4.0 - 0.6*(x/L_rho)**2.0 + 1.0)/x)/(np.pi*L_rho**3*rho_0*(1.0 - 0.75 \
           *M_OC_0*S/(np.pi*L_rho**3*rho_0*((r_OC/L_rho)**3.0*(-0.428571428571429*A_rho*(r_OC/L_rho)**4.0 - 0.6*(r_OC/L_rho)**2.0 + 1.0) - (x/L_rho)**3.0*(-0.428571428571429 \
           *A_rho*(x/L_rho)**4.0 - 0.6*(x/L_rho)**2.0 + 1.0))))*((r_OC/L_rho)**3.0*(-0.428571428571429*A_rho*(r_OC/L_rho)**4.0 - 0.6*(r_OC/L_rho)**2.0 + 1.0) - (x/L_rho)**3.0 \
           *(-0.428571428571429*A_rho*(x/L_rho)**4.0 - 0.6*(x/L_rho)**2.0 + 1.0))**2*(-np.log(1.0 - 0.75*M_OC_0*S/(np.pi*L_rho**3*rho_0*((r_OC/L_rho)**3.0*(-0.428571428571429 \
           *A_rho*(r_OC/L_rho)**4.0 - 0.6*(r_OC/L_rho)**2.0 + 1.0) - (x/L_rho)**3.0*(-0.428571428571429*A_rho*(x/L_rho)**4.0 - 0.6*(x/L_rho)**2.0 + 1.0)))) + 1.0)**2)

        return der
    
    def M_OC(self,r):
        '''Equation M_OC(t) in our paper'''
        if r==self.planet.r_OC:
            mass = 0
        else:
            mass = 4./3. * np.pi * self.planet.rho_0 * self.planet.L_rho**3 * (self.fC(self.planet.r_OC/self.planet.L_rho,self.planet.gamma)-self.fC(r/self.planet.L_rho,self.planet.gamma))
        return mass      
        
    def fC(self, r, delta): 
        '''fC (Eq. A1 Labrosse 2015)'''
        return r**3. * (1 - 3. / 5. * (delta + 1) * r**2.- 3. / 14. * (delta + 1) \
            * (2 * self.planet.A_rho - delta) * r**4.)

    def fX(self, x, r):
        '''fX (Eq. A15 Labrosse 2015)'''
        return (x)**3. * (-r**2. / 3. / self.planet.L_rho**2. + 1./5. * (1.+(r**2)/self.planet.L_rho**2.) \
                *(x)**2.-13./70. * (x)**4.) 

    def rho(self, r):
        ''' Density (Eq. 5 Labrosse 2015)'''
        return self.planet.rho_0 * (1. - r**2. / self.planet.L_rho**2. - self.planet.A_rho * r**4. / self.planet.L_rho**4.)
    
    def T_liquidus_core(self,P, S):
        '''Melting temperature (Stixrude 2014)'''
        return 6500.*(P/340.)**(0.515) * (1./(1.-np.log(1.-S)))
    
    def _S_t(self,S,r):
        if r == self.planet.r_OC:
            result = 1.
        else:
            result = S * self.M_OC(0) /self.M_OC(r) 

        if result > 1.:
            result = 1.            
        return result
     
    def _PL(self, r,P,S):
        '''Latent heat power'''
        return 4. * np.pi * r**2. * self.T_liquidus_core(P, S) * self.rho(r) * self.planet.DeltaS
        #return 4. * np.pi * r**2. * self.T_melt(r) * self.rho(r) * self.planet.DeltaS

    def _PC(self, r,P,S):
        '''Secular cooling power (Eq. A8 Labrosse 2015)'''
        return -4. * np.pi / 3. * self.planet.rho_0 * self.planet.CP * self.planet.L_rho**3. *\
                (1 - r**2. / self.planet.L_rho**2 - self.planet.A_rho* r**4. / self.planet.L_rho**4.)**(-self.planet.gamma) \
                * (self.dTL_dr_IC(r,S) + (2. * self.planet.gamma \
                * self.T_liquidus_core(P, S) * r / self.planet.L_rho**2.) *(1 + 2. * self.planet.A_rho * r**2. / self.planet.L_rho**2.) \
                /(1 - r**2. / self.planet.L_rho**2. - self.planet.A_rho * r**4. / self.planet.L_rho**4.)) \
                * (self.fC(self.planet.r_OC / self.planet.L_rho, self.planet.gamma)-self.fC(r / self.planet.L_rho, self.planet.gamma))

    def _PX(self, r):
        ''' Gravitational heat power (Eq. A14 Labrosse 2015)'''
        return 8 * np.pi**2 * self.planet.chi0 * GC * self.planet.rho_0**2 * self.planet.beta * r**2. \
        * self.planet.L_rho**2. / self.fC(self.planet.r_OC / self.planet.L_rho, 0) \
        * (self.fX(self.planet.r_OC / self.planet.L_rho, r) - self.fX(r / self.planet.L_rho, r))

    def pressure_diff(self,r):  
        '''Pressure difference (GPa)'''
        K0 = self.planet.L_rho**2/3.*2.*np.pi*GC*self.planet.rho_0**2 /1e9 #in GPa
        K0 = (2./3. * np.pi * self.planet.L_rho**2 * self.planet.rho_0**2 *GC)/1e9
        factor = (r**2)/(self.planet.L_rho**2)-(4.*r**4)/(5.*self.planet.L_rho**4)
        return -K0*factor
    
    def T_adiabat(self,r,T):
        '''Adiabatic temperature''' 
        return T*(1-r**2/self.planet.L_rho**2-self.planet.A_rho*r**4/self.planet.L_rho**4)**self.planet.gamma
    
    '''Find inner core radius when it first starts forming'''
    def find_r_IC(self,T, S):
        def Delta_T(r):
            P = self.pressure_diff(r)+self.planet.P0
            Ta = self.T_adiabat(r,T)
            TL = self.T_liquidus_core(P, S)
            return (Ta - TL)**2
        res = minimize_scalar(Delta_T, bounds=(0., 6e6), method='bounded')
        if not res.success:
            print("find_r_IC didn't converge")
        r_IC = res.x
        if r_IC < 1: r_IC = np.array(0.)
        return r_IC.tolist()
    
    def _Bc(self,rho_OC,F_th,F_X,r_IC): 
        if (F_th + F_X) < 0. or (self.planet.r_OC-r_IC) == 0.:
            Bc = 0.
        else:
            '''rms dipole field intensity at the CMB (Olson + Christensen 2006, unit:T)'''
            Bc = beta * np.sqrt(rho_OC * mu_0) * ((F_th+F_X)*self.planet.r_OC)**(1./3.)
        return Bc
    
    def _Bs (self,Bc,r_planet):
        '''rms dipole field intensity at the planetary surface, unit:T'''
        return Bc * (self.planet.r_OC/r_planet)**3 
    
        '''Magnetic moment, unit:Am2 (Olson & Christensen 2006)'''
    def _magn_moment(self,rho_OC,F_th,F_X,r_IC):
        if (F_th + F_X) < 0. or (self.planet.r_OC-r_IC) == 0:
            M = 0
        else:
            M = 4 * np.pi * self.planet.r_OC**3 * beta * np.sqrt(rho_OC/mu_0)* ((F_th + F_X)*(self.planet.r_OC-r_IC))**(1./3.)
        return M
    
    def _buoyancy_flux(self,F_th,F_X):
        '''Buoyancy flux (from Driscoll and Bercovici, eq. 35)'''
        return F_th + F_X
    
    def _F_th(self,q_cmb,qc_ad):
        '''Thermal buoyancy'''
        return self.planet.alpha_c * self.planet.gc / self.planet.rho_0 / self.planet.CP * (q_cmb - qc_ad)
    
    def _qc_ad(self,k_c,T_cmb,rho_OC):
        '''Isentropic heat flux at the CMB, unit: W m-2'''
        return k_c * T_cmb * self.planet.r_OC / (np.sqrt(3*self.planet.CP/(2*np.pi*self.planet.alpha_c*rho_OC*GC)))**2#(self.planet.r_planet)**2
    
    def _F_X(self,r,drIC_dt,S):
        '''Compositional buoyancy'''
        if S==0.:
            self.planet.Deltarho_ICB = 0.
        else:
            self.planet.Deltarho_ICB = 500./0.11 * S
        return self.planet.gc * r / self.planet.r_OC * self.planet.Deltarho_ICB /self.planet.rho_0 * (r/self.planet.r_OC)**2 * drIC_dt
    
    def _density(self,r):
        '''Planetary density'''
        return self.planet.rho_0 * (1-r**2/self.planet.L_rho**2 - self.planet.A_rho * r**4/self.planet.L_rho**4)

#class Evolution_Bouchet2013(Evolution):
#    
#    def T_liquidus_core(self,P, S):
#        return 0.
#
        
class Evolution_Labrosse2015(Evolution):    
    def T_liquidus_core(self,r):
        ''' Melting temperature (Eq. 14 Labrosse 2015)'''
        return self.planet.TL0 - self.planet.K_c * self.planet.dTL_dP * r**2. / self.planet.L_rho**2. + self.planet.dTL_dchi * self.planet.chi0 * r**3. \
                / (self.planet.L_rho**3. * self.fC(self.planet.r_OC / self.planet.L_rho, 0.))
                
    def dTL_dr_IC(self, r):
        result = -self.planet.K_c * 2.*self.planet.dTL_dP * r / self.planet.L_rho**2. \
        + 3. * self.planet.dTL_dchi * self.planet.chi0 * r**2. / (self.planet.L_rho**3. * self.fC(self.planet.r_OC / self.planet.L_rho, 0.))
        return result

    
class Rocky_Planet():
    
    def __init__(self,Mp,XFe,FeM,S):
        self.Mp = Mp
        self.XFe = XFe
        self.FeM = FeM
        self.S = S
        self.parameters(Mp,XFe,FeM)
    
    def parameters(self,Mp,XFe,FeM):
        '''Load parameter files'''
        self.read_parameters("./Ini_With_DTcmb/M_ {:.1f}_Fe_{:.0f}.0000_FeM_{:2.0f}.0000.yaml".format(Mp, XFe, FeM))
        #self.read_parameters("Earth.yaml")
        qcmb_ev = pd.read_csv("./Q_CMB/res_t_HS_Tm_Tb_qs_qc_M{:02d}_Fe{:02d}_#FeM{:02d}.res".format(int(10*Mp),int(XFe), int(FeM)), skipinitialspace=True, sep=" ", index_col=False,skiprows=[0])
        qcmb_ev.columns = ["time", "H_rad", "T_um","T_cmb","q_surf","qcmb"]
        self.time_vector = qcmb_ev["time"] *1e6
        self.qcmb = qcmb_ev["qcmb"]
            
    def read_parameters(self, file): 
        """Read parameters from yaml file"""
        with open(file, 'r') as stream:
            try:
                dict_param = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        for k, v in dict_param.items():
            setattr(self, k, float(v))
