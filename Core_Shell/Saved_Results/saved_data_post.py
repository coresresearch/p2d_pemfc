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
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
from Shared_Funcs.read_write_save_funcs import *
from Shared_Funcs.pemfc_transport_funcs import *

plt.close('all')



""" User input section """
"-----------------------------------------------------------------------------"
cwd = os.getcwd()

files = {}
#files[0] = cwd + '/folder_name'
#files[0] = cwd + '/gas_25nm'
files[0] = cwd + '/bulk_5nm'
#files[1] = cwd + '/lam_5nm'
#files[2] = cwd + '/bulk_10nm'
#files[3] = cwd + '/lam_10nm'

i_find = 2.2        # pull sv closest to this current [A/cm^2]

gas_O2y = 0         # O2 density in gas phase vs cathode depth
gas_O2cl = 0        # O2 density in gas phse vs catalyst layer depth
naf_O2r = 1         # O2 density in naf phase vs radius
naf_O2y = 1         # O2 density for inner/outter Nafion shells vs CL depth
naf_Hy = 1          # elyte potential and i_far fract vs CL depth (make sure i_cl 'on')

i_cl = 1            # fraction of i_ext vs CL depth
i_val = 0           # flux of O2 between GDL and CL vs theory i/4F
polars = 1          # polarization curves
power = 1           # power curves ontop of polarization (only use if polars = 1)
polar_val = 1       # validate polarization curve against low Pt load data [7]

# Font size on axes, legends, etc:
font_nm, font_sz = 'Arial', 14
font = plt.matplotlib.font_manager.FontProperties(family=font_nm, size=font_sz)
plt.rcParams.update({'font.size': font_sz})

# Axes spacing and limits:
#lwr_y = 0.35
#upr_y = 1.0
#y_tick = 0.1

x_lim = [0, 4]
y_lim1 = [0, 1.0]
y_lim2 = [0, 12]

# Linestyle and width:
ls = '--'
lw = 3

# Legend entries w/ optionally second:
leg_ft = 14
leg1 = ['10nm', '5nm']
leg2 = [r'Bulk-like', 'Lamellae']

# Linestyles if using second legend:
ls1 = '-'
ls2 = '--'

# Rows, columns, and subplot index:
rw = 1
cl = 3
splt_mv = 1



""" Run preprocessing to set pointers and objects """
"-----------------------------------------------------------------------------"
globals().update(ModuleReader(files[0] + '/user_inputs.csv'))
save = 0

os.chdir(files[0])
exec(open(files[0] + "/core_shell_pemfc_pre.py").read())
os.chdir(cwd)


""" Build plots for each post-processing type """
"-----------------------------------------------------------------------------"
# Generalize figure numbering
fig_num = 1

# Plot the gas-phase O2 values as a function of GDL and CL depth
if gas_O2y == 1:    
    for i in range(len(files)):
        globals().update(ModuleReader(files[0] + '/user_inputs.csv'))
        
        dat = np.genfromtxt(files[i] + '/solution.csv', delimiter=',')
        i_ind = min(range(len(dat[0,:])), key=lambda i: abs(dat[0,i]-i_find))
        sv = dat[1:,i_ind]
        
        O2_ind = gas_ca.species_index('O2')
        
        y_gdl = np.linspace(0.5*p['gdl']['dy'],t_gdl-0.5*p['gdl']['dy'],Ny_gdl)
        y_cl = np.linspace(t_gdl+0.5*p['dy'],t_gdl+t_cl-0.5*p['dy'],Ny)
        y_ca = np.hstack([y_gdl,y_cl])
        
        rho_O2_gdl = sv[iSV['rho_gdl_k'][O2_ind]::int(L_gdl/Ny_gdl)]
        rho_O2_CL = sv[iSV['rho_gas_k'][O2_ind]:L_cl:int(L_cl/Ny)]
        rho_O2_y = np.hstack([rho_O2_gdl, rho_O2_CL])
        
        plt.figure(fig_num)    
        plt.plot(y_ca*1e6, rho_O2_y, ls, linewidth=lw)
        
        plt.xlabel(r'Cathode Depth [$\mu$m]')
        plt.ylabel('Gas Phase - O2 Density [$kg/m^3$]')
        plt.legend(leg1)
        plt.tight_layout()
        
    plt.gca().set_prop_cycle(None)
    
    fig_num = fig_num + 1
    
# Plot the gas-phase O2 values as a function of only CL depth
if gas_O2cl == 1:
    for i in range(len(files)):
        globals().update(ModuleReader(files[0] + '/user_inputs.csv'))
        
        dat = np.genfromtxt(files[i] + '/solution.csv', delimiter=',')
        i_ind = min(range(len(dat[0,:])), key=lambda i: abs(dat[0,i]-i_find))        
        sv = dat[1:,i_ind]
        
        O2_ind = gas_ca.species_index('O2')
                    
        y_cl = np.linspace(t_gdl+0.5*p['dy'], t_gdl+t_cl-0.5*p['dy'],Ny)
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
    for i in range(len(files)):
        globals().update(ModuleReader(files[i] + '/user_inputs.csv')) 
        p = DictReader(files[i] + '/params.csv')
        print(t_naf, p['D_eff_naf'])
          
        dat = np.genfromtxt(files[i] + '/solution.csv', delimiter=',')
        i_ind = min(range(len(dat[0,:])), key=lambda i: abs(dat[0,i]-i_find))
        sv = dat[1:,i_ind]
        
        i_4F = dat[0,i_ind]*100**2 / (4*ct.faraday)
        print('\nO2_i_4F:', i_4F)
        
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
    clr = ['C1', 'C2', 'C3', 'C4']
    fig, ax1 = plt.subplots(fig_num)
    
    ax1.set_xlabel(r'Current Density [A/cm$^2$]', fontname=font_nm)
    ax1.set_ylabel(r'Cell Potential [V]', fontname=font_nm)
    
    if power == 1:
        ax2 = ax1.twinx()
        ax2.set_ylabel(r'Power Density [W/cm$^2$]')
    
    for i in range(len(files)):
        globals().update(ModuleReader(files[0] + '/user_inputs.csv'))
        p = DictReader(files[i] + '/params.csv')
        
        dat = np.genfromtxt(files[i] + '/solution.csv', delimiter=',')
        sv = dat[1:,:]
        
        if i == 2:
            dat = dat[:,:-2]
            sv = dat[1:,:]
                
        dphi_ss = sv[int(iSV['phi_dl']+(Ny-1)*L_cl/Ny),:] - dphi_eq_an\
                - dat[0,:]*p['R_naf'] - dat[0,:]*(0.5*p['dy'] / p['sig_naf_io']) *100**2
                
        ax1.plot(dat[0,:], dphi_ss, ls1, linewidth=lw, color=clr[i])
        
        if power == 1:
            ax2.plot(dat[0,:], dat[0,:]*dphi_ss, ls2, linewidth=lw, color=clr[i])
        
    plt.rcParams.update({'font.size': leg_ft})
#    ax1.set_ylim(y_lim1)
    
    if power == 1:
        None
#        ax2.set_ylim(y_lim2)
        
#    plt.xlim(x_lim)
    ax1.legend(leg1, loc='upper right')
    plt.tight_layout()
    
    plt.gca().set_prop_cycle(None)
    
        
    if polar_val == 1:
        x = np.array([0.003, 0.049, 0.202, 0.404, 0.803, 1.005, 1.204, 1.503, 
                      1.653, 1.851, 2.004])
        y = np.array([0.91 , 0.785, 0.724, 0.683, 0.626, 0.598, 0.572, 0.527, 
                      0.502, 0.463, 0.43 ])
        yerr = np.array([0., 0.004, 0.01 , 0.014, 0.013, 0.013, 0.019, 0.024, 
                         0.025, 0.023, 0.024])
        plt.errorbar(x, y, yerr=yerr, fmt='o', color = 'C0', capsize=3)
        
    fig_num = fig_num +1
    
# Plot Nafion-phase O2 density as a function of radius at each y-node    
if naf_O2r == 1:       
    for i in range(len(files)):
        globals().update(ModuleReader(files[0] + '/user_inputs.csv'))
        p = DictReader(files[i] + '/params.csv')
        
        legend_str = []
        legend_count = 0
        
        dat = np.genfromtxt(files[i] + '/solution.csv', delimiter=',')
        i_ind = min(range(len(dat[0,:])), key=lambda i: abs(dat[0,i]-i_find))
        sv = dat[1:,i_ind]

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
        if i == 0:
            plt.ylabel('Nafion Phase - O2 Density *1e4 [$kg/m^3$]')
        elif i == len(files) -1:
            plt.rcParams.update({'font.size': leg_ft})
            plt.legend(legend_str, loc='lower right')
            plt.rcParams.update({'font.size': ft})
            
        plt.ylim(y_lim1)        
        
        plt.gca().set_prop_cycle(None)
        
    fig_num = fig_num +1
    
if naf_O2y == 1:
    clr = ['C1', 'C2', 'C3', 'C4']
    fig, ax1 = plt.subplots(num=fig_num)
    
    ax1.set_xlabel(r'Catalyst layer depth [$\mu$m]', fontname=font_nm)
    ax1.set_ylabel(r'O$_2$(Naf) Density *1e3 [kg/m$^3$]', fontname=font_nm)
    
    ax2 = ax1.twinx()
    
    for i in range(len(files)):
        globals().update(ModuleReader(files[0] + '/user_inputs.csv'))
        p = DictReader(files[i] + '/params.csv')
        
        dat = np.genfromtxt(files[i] + '/solution.csv', delimiter=',')
        i_ind = min(range(len(dat[0,:])), key=lambda i: abs(dat[0,i]-i_find))
        sv = dat[1:,i_ind]
        
        rho_O2_out_y = np.flip(sv[int(iSV['rho_n1']+p['Nr']-1):L_cl:int(L_cl/Ny)], 0)
        ax1.plot(np.linspace(0,t_cl*1e6,Ny), rho_O2_out_y*1e3, ls1, linewidth=lw, color=clr[i])
        
        rho_O2_in_y = np.flip(sv[int(iSV['rho_n1']):L_cl:int(L_cl/Ny)], 0)
        ax2.plot(np.linspace(0,t_cl*1e6,Ny), rho_O2_in_y*1e3, ls2, linewidth=lw, color=clr[i])
        ax2.get_yaxis().set_visible(False)
    
    plt.rcParams.update({'font.size': leg_ft})
#    ax1.set_ylim(y_lim1)
#    ax2.set_ylim(y_lim1)
#    plt.xlim(x_lim)
    ax1.legend(leg1, loc='lower right')
    plt.tight_layout()
        
if i_cl == 1:
    i_Far_cl = np.zeros([len(files), p['Ny']])
            
    for i in range(len(files)):
        globals().update(ModuleReader(files[0] + '/user_inputs.csv'))
        p = DictReader(files[i] + '/params.csv')
        
        legend_str = []
        legend_count = 0
        
        dat = np.genfromtxt(files[i] + '/solution.csv', delimiter=',')
        i_ind = min(range(len(dat[0,:])), key=lambda i: abs(dat[0,i]-i_find))
        i_ext = dat[0,i_ind]*100**2
        sv = dat[1:,i_ind]
                
        # Set up storage location and CL depths:
        y_cl = np.linspace(t_gdl+0.5*p['dy'], t_gdl+t_cl-0.5*p['dy'],Ny)
        
        # Start with the first node (y: GDL -> Elyte, r: in -> out)
        sv_mv = 0 
        sv_nxt = int(L_cl*p['1/Ny'])
        
        for j in range(p['Ny']):            
            # Set the inner shell state:---------------------------------------
            phi_dl = sv[iSV['phi_dl']+sv_mv]
            carb_ca.electric_potential = 0
            pt_s_ca.electric_potential = 0
        
            naf_b_ca.electric_potential = -phi_dl
            naf_s_ca.electric_potential = -phi_dl
        
            rho_naf_k = np.hstack((p['rho_H'], sv[iSV['rho_n1']+sv_mv]))
        
            T_ca = sv[iSV['T']+sv_mv]
            naf_b_ca.TDY = T_ca, sum(rho_naf_k), rho_naf_k
        
            # i_Far [A/cm^2] at each CL depth:---------------------------------
            sdot_e = pt_s_ca.net_production_rates[p['ind_e']]
            i_Far_cl[i, j] = sdot_e*ct.faraday*p['SA_pv_pt']
            
            sv_mv = sv_mv + sv_nxt
            
        plt.figure(fig_num)
        plt.plot(y_cl*1e6, -i_Far_cl[i,:] / (i_ext*p['1/dy']), ls, linewidth=lw)
        
        plt.xlabel(r'Cathode Depth [$\mu$m]')
        plt.ylabel('$i_{Far}$ / $i_{ext}$ [-]')
        plt.legend(leg1)
        plt.tight_layout()
        
    plt.gca().set_prop_cycle(None)
        
    fig_num = fig_num +1
    
if naf_Hy == 1:
    clr = ['C1', 'C2', 'C3', 'C4']
    fig, ax1 = plt.subplots(num=fig_num)
    
    ax1.set_xlabel(r'Catalyst layer detpth [$\mu$m]', fontname=font_nm)
    ax1.set_ylabel(r'Electrolyte Potential [V]', fontname=font_nm)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'i$_{Far}$ / i$_{ext}$ [-]', fontname=font_nm)
    
    for i in range(len(files)):
        globals().update(ModuleReader(files[0] + '/user_inputs.csv'))
        p = DictReader(files[i] + '/params.csv')
        
        dat = np.genfromtxt(files[i] + '/solution.csv', delimiter=',')
        i_ind = min(range(len(dat[0,:])), key=lambda i: abs(dat[0,i]-i_find))
        sv = dat[1:,i_ind]
        
        phi_elyte = np.flip(-1*sv[int(iSV['phi_dl']):L_cl:int(L_cl/Ny)], 0)
        ax1.plot(np.linspace(0,t_cl*1e6,Ny), phi_elyte, ls1, linewidth=lw, color=clr[i])
        
        i_far_frac = np.flip(-i_Far_cl[i,:] / (i_ext*p['1/dy']), 0)
        ax2.plot(np.linspace(0,t_cl*1e6,Ny), i_far_frac, ls2, linewidth=lw, color=clr[i])
        
    plt.rcParams.update({'font.size': leg_ft})
#    ax1.set_ylim(y_lim1)
#    ax2.set_ylim(y_lim2)
#    plt.xlim(x_lim)
    ax1.legend(leg1)
    plt.tight_layout()
        
""" Add second legend if needed to emphasize line style """
if leg2 is not None:
    from matplotlib.lines import Line2D
    lines = [Line2D([0], [0], color='k', linestyle=ls1), 
             Line2D([0], [0], color='k', linestyle=ls2)]
    plt.rcParams.update({'font.size': leg_ft})
    plt.legend(lines, leg2, loc='lower left')
    ax2.add_artist(leg1)
    plt.rcParams.update({'font.size': ft})
