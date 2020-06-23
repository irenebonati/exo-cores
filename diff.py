#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:13:54 2020

@author: irenebonati
"""
import sympy as sp
import numpy as np

''' Melting temperature jump at ICB '''
'''Stixrude'''

x = sp.symbols('x')
K0 = sp.symbols('K0')
P0 = sp.symbols('P0')
L_rho = sp.symbols('L_rho')
S = sp.symbols('S')
rho_0 = sp.symbols('rho_0')
r_OC = sp.symbols('r_OC')
gamma = sp.symbols('gamma')
M_OC_0 = sp.symbols('M_OC_0')
A_rho = sp.symbols('A_rho')
a = sp.symbols('a')
c = sp.symbols('c')
T0 = sp.symbols('T0')

P = P0 - K0 * ((x**2)/(L_rho**2) - (4.*x**4)/(5.*L_rho**4))

fC1 = (r_OC/L_rho)**3. * (1. - 3. / 5. * (0. + 1.) * (r_OC/L_rho)**2.- 3. / 14. * (0. + 1)* (2 * A_rho - 0.) * (r_OC/L_rho)**4.)

fC2 = (x/L_rho)**3. * (1. - 3. / 5. * (0. + 1.) * (x/L_rho)**2.- 3. / 14. * (0. + 1.) * (2 * A_rho - 0.) * (x/L_rho)**4.)

S = (S * M_OC_0) /(4./3. * sp.pi * rho_0 * L_rho**3 * (fC1-fC2))

function = 6500. * (P/340.)**(0.515) * 1./(1.-sp.log(1.-S))

#der = sp.diff(function,x)
#print (der)

## Bouchet differentiation

P = P0 - K0 * ((x**2)/(L_rho**2) - (4.*x**4)/(5.*L_rho**4))

function_Bouchet = ((P-P0)/a+1.)**(1./c) * T0
der = sp.diff(function_Bouchet,x)
print ('Bouchet=',der)


