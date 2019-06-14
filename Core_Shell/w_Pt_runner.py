# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:33:25 2019

@author: Corey Randall
"""

import pylab as plt
import numpy as np

plt.close('all')

w_Pt_vec = np.array([0.2, 0.1, 0.05, 0.025])
" Loop through Pt loading polarization curves: "
#cti = ['pemfc2.cti', 'pemfc1.cti', 'pemfc05.cti', 'pemfc025.cti']
#for i, w in enumerate(w_Pt_vec):
#    w_Pt = w
#    ctifile = cti[i]
#    exec(open("particle_shell_pemfc_runner.py").read())

" Check how sig_io changes as a function of w_Pt: "
" sig_naf_io_func(temp, t_naf, RH, p_Pt, p, method) "
#sig_io = np.zeros_like(w_Pt_vec)
#for ind, w in enumerate(w_Pt_vec):
#    w_Pt = w
#    exec(open("particle_shell_pemfc_pre.py").read())
#    sig_io[ind] = p['sig_naf_io']
#    
#plt.plot(w_Pt_vec, sig_io)

" Plot low Pt curves for ECS Presentation and properties vs w_Pt "
#font_nm, font_sz = 'Arial', 14
#font = plt.matplotlib.font_manager.FontProperties(family=font_nm, size=font_sz)
#plt.rcParams.update({'font.size': font_sz})
#clr = ['C0', 'C1', 'C2', 'C3']

#w_Pt_vec = np.linspace(0.01,0.4,50)
#sig_io = np.zeros_like(w_Pt_vec)
#D_o2 = np.zeros_like(w_Pt_vec)
#for ind, w in enumerate(w_Pt_vec):
#    w_Pt = w
#
#    method = 'mix'
#    sig_method = method
#    D_O2_method = method
#    theta = 45
#    
#    exec(open("particle_shell_pemfc_pre.py").read())
#    sig_io[ind] = p['sig_naf_io']
#    D_o2[ind] = p['D_eff_naf']
#    
#fig, ax1 = plt.subplots()
#ax1.set_ylabel(r'Effective Conductivity [S/m]', fontname=font_nm)
#ax1.set_xlabel(r'Pt Loading [mg/cm$^2$]', fontname=font_nm)
#ax1.plot(w_Pt_vec, sig_io, linewidth=2, color='C0')
#ax1.set_ylim([0.8, 1.4])
#
#ax2 = ax1.twinx()
#ax2.set_ylabel(r'Effective O$_2$ Diffusion Cofficient [m$^2$/s]', fontname=font_nm)
#ax2.plot(w_Pt_vec, D_o2, linewidth=2, color='C0', linestyle='--')
#ax2.set_ylim([2e-12, 6e-12])
#plt.tight_layout()
    
#for i, w in enumerate(w_Pt_vec):
#    w_Pt = w
#    
#    plt.ylabel(r'Cell Voltage [V]', fontname=font_nm)
#    plt.xlabel(r'Current Density [A/cm$^2$]', fontname=font_nm)
#    plt.tight_layout()
#    
#    if w_Pt == 0.2:
#        x = np.array([0.008, 0.051, 0.201, 0.403, 0.802, 1.002, 1.202, 1.501, 
#                      1.651, 1.851, 2.000])
#        y = np.array([0.952, 0.849, 0.803, 0.772, 0.731, 0.716, 0.700, 0.675, 
#                      0.665, 0.647, 0.634])
#        yerr = np.array([0, 0.012, 0.007, 0.007, 0.012, 0.001, 0.008, 0.007,
#                         0.007, 0.009, 0.009])
#        color, fmt, lbl = 'black', '-x', r'0.2 mg/cm$^$2'
#    elif w_Pt == 0.1:
#        x = np.array([0.006, 0.053, 0.201, 0.401, 0.802, 1.002, 1.200, 1.499, 
#                      1.651, 1.851, 2.000])
#        y = np.array([0.930, 0.834, 0.785, 0.754, 0.711, 0.691, 0.673, 0.649, 
#                      0.635, 0.615, 0.598])
#        yerr = np.array([0, 0.009, 0.007, 0.005, 0.007, 0.011, 0.011, 0.007, 
#                         0.009, 0.011, 0.011])
#        color, fmt, lbl = 'black', '-o', r'0.1 mg/cm$^$2'
#    elif w_Pt == 0.05:
#        x = np.array([0.008, 0.053, 0.201, 0.401, 0.800, 1.000, 1.200, 1.500, 
#                      1.651, 1.850, 2.001])
#        y = np.array([0.919, 0.810, 0.760, 0.724, 0.674, 0.653, 0.634, 0.603,
#                      0.585, 0.558, 0.537])
#        yerr = np.array([0, 0.008, 0.006, 0.006, 0.007, 0.007, 0.005, 0.005, 
#                         0.006, 0.007, 0.007])
#        color, fmt, lbl = 'black', '-s', r'0.05 mg/cm$^$2'
#    elif w_Pt == 0.025:
#        x = np.array([0.003, 0.049, 0.202, 0.404, 0.803, 1.005, 1.204, 1.503, 
#                      1.653, 1.851, 2.004])
#        y = np.array([0.910, 0.785, 0.724, 0.683, 0.626, 0.598, 0.572, 0.527, 
#                      0.502, 0.463, 0.430])
#        yerr = np.array([0, 0.004, 0.010, 0.014, 0.013, 0.013, 0.019, 0.024, 
#                         0.025, 0.023, 0.024])
#        color, fmt, lbl = 'black', '-^', r'0.025 mg/cm$^$2'
#        
#    try:
#        plt.errorbar(x, y, yerr=yerr, fmt='.', mfc='none', linewidth=2, 
#                     color=clr[i], capsize=3, label=lbl)
#        
#        plt.ylim([0.35, 1.0])
#        plt.xlim([0, 2.1])
#    except:
#        None
#
#plt.rcParams.update({'font.size': font_sz -4})
#plt.legend([r'Pt = 0.2 mg/cm$^2$', r'Pt = 0.1  mg/cm$^2$', r'Pt = 0.05  mg/cm$^2$',
#            r'Pt = 0.025  mg/cm$^2$'], loc='lower left')

" Extra resistance vs w_Pt "
#R = np.array([5, 10, 25, 45])
#x2 = np.linspace(0.025, 0.2, 50)
#
#def f(x):
#    f = 0.8736*x**(-1.083)
#    return f
#
#plt.plot(w_Pt_vec, R, 'o', color='k')
#plt.plot(x2, f(x2), '--', linewidth=2, color='k')
#
#plt.xlim([0, 0.25])
#plt.ylim([0, 50])
#
#plt.xlabel(r'Pt loading [mg/cm$^2$]')
#plt.ylabel(r'Resistance [m$\Omega$-cm$^2$]')
#plt.tight_layout()

" Exchange current density vs w_Pt "
i_o = np.array([4, 2, 0.3, 0.05]) *1e10
x2 = np.linspace(0.025, 0.2, 50)

def f(x):
    f = 23.426*x - 0.6087
    return f *1e10

plt.plot(w_Pt_vec, i_o, 'o', color='k')
plt.plot(x2, f(x2), '--', linewidth=2, color='k')

plt.xlim([0, 0.25])
plt.xticks(np.arange(0, 0.25, 0.1))
plt.ylim([0, 5*1e10])

plt.xlabel(r'Pt loading [mg/cm$^2$]')
plt.ylabel(r'$i^*_o$ [A/m$^2$*(m$^3$/kmol)$^{6.5}$]')
plt.tight_layout()