# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:33:47 2019

@author: Corey Randall
"""

import numpy as np
import pylab as plt
from pemfc_property_funcs import D_eff_naf_func, sig_naf_io_func

plt.close('all')

# Font size on axes, legends, etc:
plt.rcParams.update({'font.size': 18})

t = np.linspace(5,100,50)
D = np.zeros_like(t)
sig = np.zeros_like(t)

for i,e in enumerate(t):
    D[i] = D_eff_naf_func(333, t[i]*1e-9, 'lam')
    sig[i] = sig_naf_io_func(333, t[i]*1e-9, 95, 'lam')
    
plt.figure(1)
plt.plot(t,D,linewidth=3)
plt.xlabel('Nafion Thickness [nm]')
plt.ylabel(r'$D_{O2,eff}$ [m$^{2}$/s]')
plt.tight_layout()

plt.figure(2)
plt.plot(t,sig,linewidth=3)
plt.xlabel('Nafion Thickness [nm]')
plt.ylabel(r'$\sigma_{io,eff}$ [S/m]')
plt.tight_layout()