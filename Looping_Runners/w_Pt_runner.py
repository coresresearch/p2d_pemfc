"""
Created on Mon May 20 15:33:25 2019

Author: Corey R. Randall

Notes:
    This script allows the PEMFC models to be run while looping over different
    Pt loading values. Additional sections of the script produce plots that 
    were used in a presentation given at the 235th ECS conference in Dallas, TX.
    Since this script is only meant to loop through Pt loading values, all other
    desired user inputs should be specified in the file pemfc_runner.py.
    
    Read through the string comments for each section and uncomment the code 
    below it in order to produce its outputs.
"""

import os
import pylab as plt
import numpy as np

os.chdir(os.getcwd() + '/../')
plt.close('all')

font_nm, font_sz = 'Arial', 14
font = plt.matplotlib.font_manager.FontProperties(family=font_nm, size=font_sz)
plt.rcParams.update({'font.size': font_sz})

" Loop through Pt loading polarization curves: "
folder_nm_generic = 'cs_Pt_loop_' # pre-fix to all folders for saving

y_model = np.array([])
w_Pt_vec = np.array([0.2, 0.1, 0.05, 0.025]) # Pt-loading values to loop through
for i, w in enumerate(w_Pt_vec):
    from pemfc_runner import *
    import pemfc_runner as user_inputs
    
    save = user_inputs.save = 0 # change to one if wanting to save outputs
    w_Pt = user_inputs.w_Pt = w
                
    folder_name = user_inputs.folder_name = folder_nm_generic + str(w_Pt) + 'mgcm-2'
    exec(open("Shared_Funcs/pemfc_pre.py").read())
    exec(open("Shared_Funcs/pemfc_dsvdt.py").read())
    exec(open("Shared_Funcs/pemfc_post.py").read())
    y_model = np.hstack([y_model, dphi_ss[1:]])
    if save == 1:
        ModuleWriter(cwd +'/Saved_Results/' +folder_name +'/user_inputs.csv', user_inputs)

" Plot sig_io and D_O2 as f(w_Pt) for 'mix' method "
#clr = ['C0', 'C1', 'C2', 'C3']
#
#from pemfc_runner import *
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
#    exec(open("Shared_Funcs/pemfc_pre.py").read())
#    sig_io[ind] = cl['sig_naf_io']
#    D_o2[ind] = cl['D_eff_naf']
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

" Recreate low Pt loading curve from data only showing performance losses "
#w_Pt_vec = np.array([0.2, 0.1, 0.05, 0.025])
#for i, w in enumerate(w_Pt_vec):
#    w_Pt = w
#    
#    plt.ylabel(r'Cell Voltage [V]', fontname=font_nm)
#    plt.xlabel(r'Current Density [A/cm$^2$]', fontname=font_nm)
#    plt.tight_layout()
#    
#    if w_Pt == 0.2:
#        x = np.array([0.000, 0.050, 0.200, 0.400, 0.800, 1.000, 1.200, 1.500, 
#                      1.650, 1.850, 2.000])
#        y = np.array([0.952, 0.849, 0.803, 0.772, 0.731, 0.716, 0.700, 0.675, 
#                      0.665, 0.647, 0.634])
#        yerr = np.array([0, 0.012, 0.007, 0.007, 0.012, 0.001, 0.008, 0.007,
#                         0.007, 0.009, 0.009])
#        color, fmt, lbl = 'black', '-x', r'0.2 mg/cm$^$2'
#    elif w_Pt == 0.1:
#        x = np.array([0.000, 0.050, 0.200, 0.400, 0.800, 1.000, 1.200, 1.500, 
#                      1.650, 1.850, 2.000])
#        y = np.array([0.930, 0.834, 0.785, 0.754, 0.711, 0.691, 0.673, 0.649, 
#                      0.635, 0.615, 0.598])
#        yerr = np.array([0, 0.009, 0.007, 0.005, 0.007, 0.011, 0.011, 0.007, 
#                         0.009, 0.011, 0.011])
#        color, fmt, lbl = 'black', '-o', r'0.1 mg/cm$^$2'
#    elif w_Pt == 0.05:
#        x = np.array([0.000, 0.050, 0.200, 0.400, 0.800, 1.000, 1.200, 1.500, 
#                      1.650, 1.850, 2.000])
#        y = np.array([0.919, 0.810, 0.760, 0.724, 0.674, 0.653, 0.634, 0.603,
#                      0.585, 0.558, 0.537])
#        yerr = np.array([0, 0.008, 0.006, 0.006, 0.007, 0.007, 0.005, 0.005, 
#                         0.006, 0.007, 0.007])
#        color, fmt, lbl = 'black', '-s', r'0.05 mg/cm$^$2'
#    elif w_Pt == 0.025:
#        x = np.array([0.000, 0.050, 0.200, 0.400, 0.800, 1.000, 1.200, 1.500, 
#                      1.650, 1.850, 2.000])
#        y = np.array([0.910, 0.785, 0.724, 0.683, 0.626, 0.598, 0.572, 0.527, 
#                      0.502, 0.463, 0.430])
#        yerr = np.array([0, 0.004, 0.010, 0.014, 0.013, 0.013, 0.019, 0.024, 
#                         0.025, 0.023, 0.024])
#        color, fmt, lbl = 'black', '-^', r'0.025 mg/cm$^$2'
#    try:
#        plt.errorbar(x, y, yerr=yerr, fmt=fmt, mfc='none', linewidth=1,
#                     color=color, capsize=3, label=lbl)
#        
#        plt.ylim([0.35, 1.0])
#        plt.xlim([0, 2.1])
#    except:
#        None
#
#plt.rcParams.update({'font.size': font_sz -4})
#plt.legend([r'Pt = 0.2 mg/cm$^2$', r'Pt = 0.1  mg/cm$^2$', r'Pt = 0.05  mg/cm$^2$',
#            r'Pt = 0.025  mg/cm$^2$'], loc='lower left')

" Extra resistance as f(w_Pt) - attempt 1 at better fits "
#R = np.array([5, 10, 25, 45])
#x2 = np.linspace(0.025, 0.2, 50)
#w_Pt_vec = np.array([0.2, 0.1, 0.05, 0.025])
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
#plt.xlabel(r'Pt loading [mg/cm$^2$]', fontname=font_nm)
#plt.ylabel(r'Resistance [m$\Omega$-cm$^2$]', fontname=font_nm)
#plt.tight_layout()

" Exchange current density as f(w_Pt) - attempt 2 at better fits "
#w_Pt_vec = np.array([0.2, 0.1, 0.05, 0.025])
#i_o = np.array([4, 2, 0.3, 0.05]) *1e10
#x2 = np.linspace(0.025, 0.2, 50)
#
#def f(x):
#    f = 23.426*x - 0.6087
#    return f *1e10
#
#plt.plot(w_Pt_vec, i_o, 'o', color='k')
#plt.plot(x2, f(x2), '--', linewidth=2, color='k')
#
#plt.xlim([0, 0.25])
#plt.xticks(np.arange(0, 0.25, 0.1))
#plt.ylim([0, 5*1e10])
#
#plt.xlabel(r'Pt loading [mg/cm$^2$]', fontname=font_nm)
#plt.ylabel(r'$i^*_o$ [A/m$^2$*(m$^3$/kmol)$^{6.5}$]', fontname=font_nm)
#plt.tight_layout()