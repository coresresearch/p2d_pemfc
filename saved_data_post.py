"""
Post-processing script:
    This script was written in order to generate results from saved solution
    vectors created from the particle-shell model. It is capable of reading in
    the solution.csv file along with relavent parameters. Preprocessing is 
    performed if needed based on these files and the resulting information is 
    then manipulated to create plots or perform calculations to help validate 
    the model.
"""

""" Import needed modules """
"-----------------------------------------------------------------------------"
import os, ast
import numpy as np
import pylab as plt
import cantera as ct
from pemfc_transport_funcs import *



""" User input section """
"-----------------------------------------------------------------------------"
cwd = os.getcwd()

files = {}
files[0] = cwd + '/gas_25nm'
files[1] = cwd + '/gas_10nm'
files[2] = cwd + '/gas_5nm'

i_find = 0.1        # pull sv closest to this current [A/cm^2]

gas_O2y = 1         # O2 density in gas phase vs cathode depth
gas_O2cl = 1        # O2 density in gas phse vs catalyst layer depth
naf_O2r = 1         # O2 density in naf phase vs radius

i_cl = 0            # fraction of i_ext vs CL depth
i_val = 1           # flux of O2 between GDL and CL vs theory i/4F
polars = 1          # polarization curves

# Font size on axes, legends, etc:
plt.rcParams.update({'font.size': 12})

# Linestyle and width:
ls = '--'
lw = 2

# Legend entries w/ optionally second:
leg1 = ['25nm', '10nm', '5nm']
leg2 = None #['carb', 'gas']

# Linestyles if using second legend:
ls1 = '-'
ls2 = '--'

# Rows, columns, and subplot index:
rw = 1
cl = 3
splt_mv = 1



""" Run preprocessing to set pointers and objects """
"-----------------------------------------------------------------------------"
from particle_shell_pemfc_runner import *
save = 0

exec(open("particle_shell_pemfc_pre.py").read())



""" Build plots for each post-processing type """
"-----------------------------------------------------------------------------"
# Generalize figure numbering
fig_num = 1

# Plot the gas-phase O2 values as a function of GDL and CL depth
if gas_O2y == 1:    
    for i in range(len(files)):
        dat = np.genfromtxt(files[i] + '/solution.csv', delimiter=',')
        i_ind = min(range(len(dat[0,:])), key=lambda i: abs(dat[0,i]-i_find))
        sv = dat[1:,i_ind]
        
        O2_ind = gas_ca.species_index('O2')
        
        y_gdl = np.linspace(0.5*p['gdl']['dy'],t_gdl-0.5*p['gdl']['dy'],Ny_gdl)
        y_cl = np.linspace(t_gdl+0.5*p['dy'],t_gdl+t_ca-0.5*p['dy'],Ny)
        y_ca = np.hstack([y_gdl,y_cl])
        
        rho_O2_gdl = sv[iSV['rho_gdl_k'][O2_ind]::int(L_gdl/Ny_gdl)]
        rho_O2_CL = sv[iSV['rho_gas_k'][O2_ind]:L_cl:int(L_cl/Ny)]
        rho_O2_y = np.hstack([rho_O2_gdl, rho_O2_CL])
        
        plt.figure(fig_num)    
        plt.plot(y_ca*1e6, rho_O2_y, ls, linewidth=lw)
        
        plt.xlabel(r'Cathode Depth [$\mu$m]')
        plt.ylabel('Gas Phase - O2 Density [$kg/m^3$]')
        plt.tight_layout()
        
    plt.gca().set_prop_cycle(None)
    
    fig_num = fig_num + 1
    
# Plot the gas-phase O2 values as a function of only CL depth
if gas_O2cl == 1:
    for i in range(len(files)):
        dat = np.genfromtxt(files[i] + '/solution.csv', delimiter=',')
        i_ind = min(range(len(dat[0,:])), key=lambda i: abs(dat[0,i]-i_find))        
        sv = dat[1:,i_ind]
        
        O2_ind = gas_ca.species_index('O2')
                    
        y_cl = np.linspace(t_gdl+0.5*p['dy'], t_gdl+t_ca-0.5*p['dy'],Ny)
        rho_O2_CL = sv[iSV['rho_gas_k'][O2_ind]:L_cl:int(L_cl/Ny)]
        
        plt.figure(fig_num)
        plt.plot(y_cl*1e6, rho_O2_CL, ls, linewidth=lw)
        
        plt.xlabel(r'Cathode Depth [$\mu$m]')
        plt.ylabel('Gas Phase - O2 Density [$kg/m^3$]')
        plt.tight_layout()
        
    plt.gca().set_prop_cycle(None)
    
    fig_num = fig_num + 1

# Calculate the flux between GDL and CL and validate to i/4F    
if i_val == 1:
    i_4F = i_find*100**2 / (4*ct.faraday)
    print('\nO2_i_4F:', i_4F)
    
    for i in range(len(files)):
        
        f = open(files[i] + '/params.csv', 'r')
        reader = csv.reader(f)
        
        p = {}
        for row in reader:
            k, v = row
            
            if '[' not in v:
                p[k] = eval(v)
            else:
                p[k] = " ".join(v.split()).replace(' ',', ')
                p[k] = np.asarray(eval(p[k]))
        
        f.close()
                   
        dat = np.genfromtxt(files[i] + '/solution.csv', delimiter=',')
        i_ind = min(range(len(dat[0,:])), key=lambda i: abs(dat[0,i]-i_find))
        sv = dat[1:,i_ind]
        
        rho_gdl_k = sv[iSV['rho_gdl_k']+int(L_gdl/Ny_gdl*(Ny_gdl-1))]
        TDY1 = sv[iSV['T_gdl']], sum(rho_gdl_k), rho_gdl_k
        
        rho_cl_k = sv[iSV['rho_gas_k']]
        TDY2 = sv[iSV['T']], sum(rho_cl_k), rho_cl_k
        
        O2_BC_flux = fickian_adf(TDY1, TDY2, gas_ca, p['gdl_cl'], 1)\
                   / gas_ca.molecular_weights
        
        print('file:', i, ' - i_ext:', dat[0,i_ind],
              '- O2_BC_flux:', O2_BC_flux[0], 'ratio:', i_4F /O2_BC_flux[0])
        
# Plot all polarization curves on the same figure
if polars == 1:   
    for i in range(len(files)):
        dat = np.genfromtxt(files[i] + '/solution.csv', delimiter=',')
        sv = dat[1:,:]
        
        dphi_ss = sv[int(iSV['phi_dl']+(Ny-1)*L_cl/Ny),:] - dphi_eq_an
        
        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111)
        plt.plot(np.hstack(dat[0,:]), dphi_ss, ls, linewidth=lw)
    
        plt.ylabel(r'Steady-state Potential, $\Delta \Phi$ [V]')
        plt.xlabel(r'External Current, $i_{ext}$ [$A/cm^2$]')
        leg = plt.legend(leg1)
        plt.tight_layout()
    
    plt.gca().set_prop_cycle(None)
        
    fig_num = fig_num +1
    
# Plot Nafion-phase O2 density as a function of radius at each y-node    
if naf_O2r == 1:       
    for i in range(len(files)):
        legend_str = []
        legend_count = 0
        
        dat = np.genfromtxt(files[i] + '/solution.csv', delimiter=',')
        i_ind = min(range(len(dat[0,:])), key=lambda i: abs(dat[0,i]-i_find))
        sv = dat[1:,i_ind]
        
        f = open(files[i] + '/params.csv', 'r')
        reader = csv.reader(f)
        
        p = {}
        for row in reader:
            k, v = row
            
            if '[' not in v:
                p[k] = eval(v)
            else:
                p[k] = " ".join(v.split()).replace(' ',', ')
                p[k] = np.asarray(eval(p[k]))
        
        f.close()

        for j in range(Ny):
            ind_1 = int(iSV['rho_n1']+j*L_cl/Ny)
            ind_2 = int(iSV['rho_n1']+j*L_cl/Ny + Nr)
            rho_k_r = sv[ind_1:ind_2]
    
            plt.figure(fig_num)
            plt.subplot(rw,cl,i+splt_mv)
            plt.plot(1/p['1/r_j']*1e9, rho_k_r*1e4, ls, linewidth=lw)
            legend_str.append('y node = ' + str(legend_count))
            legend_count = legend_count +1
    
        plt.xlabel('Nafion Shell Radius, [nm]')
        plt.ylabel('Nafion Phase - O2 Density *1e4 [$kg/m^3$]')
        plt.legend(legend_str, loc='lower right')
        
        plt.gca().set_prop_cycle(None)
        
    fig_num = fig_num +1
    
if i_cl == 1: 
    print()
    
""" Copy paste to add second legend if plotting carbon and gas on the same 
figure together. Do this after generating the regular polarization. """
if leg2 is not None:
    from matplotlib.lines import Line2D
    lines = [Line2D([0], [0], color='k', linestyle=ls1), 
             Line2D([0], [0], color='k', linestyle=ls2)]
    plt.legend(lines, leg2, loc='lower left')
    ax.add_artist(leg)
