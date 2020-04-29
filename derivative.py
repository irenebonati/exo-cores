#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:50:33 2020

@author: irenebonati
"""

# Differentiation with python

import sympy as sp

P0, K0, L, S, r = sp.symbols('P0 K0 L S r')

P = P0 - K0 * (r**2/L**2 - (4*r**4)/(5*L**4))

function = 6500 * (P/340)**(0.515) * 1./(1-sp.log(1-S))

der = sp.diff(function,r)

test = sp.diff(r**2)

print (der)
print (test)
