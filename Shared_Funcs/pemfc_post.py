""" File Description """
"-----------------------------------------------------------------------------"
"""This file is used for post processing data from the pemfc_dsvdt.py file. 
The afforementioned file runs a pseudo 2D model of a PEM fuel cell using one of
two geometries for the catalyst layer: core-shell or flooded-agglomerate. A 
matrix of steady-state data from the model stores: the external currents, gas 
phase densities, electrolyte phase densities, Pt surface coverages, temperatures,
and double layer potentials. Using this information, calculations can be done in
order to provide useful information about the predicted fuel cell performance
based on the model.

Flags in the pemfc_runner.py file determine which parts of the post processing
are performed. 'debug' allows the user to see the double layer potential, Nafion
species densities, gas phase species densities, and Pt surface coverages all 
plotted against time. Primarily this is used as a debugging tool so that the user
can make sure all variables are reaching steady-state. 'grads' produces plots 
for double layer potential, O2 in the gas phase, O2 in the Nafion phase, and the
fraction of Faradaic current vs the catalyst layer depth. 'radial' plots the O2
density in the Nafion phase against the radius of the shells or agglomerate. 
'over_p' and 'polar' produce an overpotential and polarization curve respectively.
'i_ver' takes a selected current from pemfc_runner.py and calulates the O2 flux
at the GDL/CL boundary using theory (i_ext / 4*F) and Fick's law. These values
are compared in a ratio (which should be equal to ~1) to provide more evidence
to help verify the model is working properly and meeting the corrent boundary
conditions.

From pemfc_runner.py if 'save' is turned on, any of the plots produced from this
post processing file will be saved along with the previously mentioned matrix of
results."""

""" Import needed modules """
"-----------------------------------------------------------------------------"
import os
import numpy as np
import pylab as plt
from Shared_Funcs.read_write_save_funcs import *

""" Post-processing for Plotting and Other Results """
"-----------------------------------------------------------------------------"
# Update plot settings:
plt.close('all')
font = plt.matplotlib.font_manager.FontProperties(family=font_nm, size=font_sz)
plt.rcParams.update({'font.size': font_sz})

# Move into correct directory to save any outputs:
cwd = os.getcwd()

if save:
    import pemfc_runner as user_inputs
    try:
        os.chdir(cwd + '/Saved_Results')
    except:
        os.mkdir(cwd + '/Saved_Results')  
        os.chdir(cwd + '/Saved_Results')
        
    SaveFiles(folder_name, ctifile, p, sv_save, user_inputs)
    os.chdir(cwd + '/Saved_Results/' + folder_name)
    
# Generalize figure numbering:
fig_num = 1

# Species indexes:
iO2_g = gas_ca.species_index('O2')
iO2_n = naf_b_ca.species_index('O2(Naf)')
iH_n = naf_b_ca.species_index('H(Naf)')

# Toggle for shell indecies if using flooded_agg:
if model == 'core_shell': start_tog, shl_tog = 'rho_naf_k', 0
elif model == 'flooded_agg': start_tog, shl_tog = 'rho_shl_k', 1

if debug:
    # If turned 'on' in pemfc_runner.py, this flag will produce plots for double
    # layer voltage, gas-phase species, Nafion species, and Pt coverages to 
    # help debug the code and ensure that all solution vector variables are 
    # reaching steady-state for the user-specified time form pemfc_runner.py.
    
    # Extract cathode double layer potential and plot:
    phi_dl = sol.y[iSV['phi_dl'], :]

    plt.figure(fig_num)
    plt.plot(sol.t, phi_dl)

    plt.ylabel(r'Cathode Double Layer Voltage [V]')
    plt.xlabel('Time, t [s]')
    plt.tight_layout()
    
    if save:
        plt.savefig('Double_Layer_v_Time.png')
        
    fig_num = fig_num +1
    
    for i in range(Ny_cl):
        # Extract nafion species densities and plot:
        legend_str = []
        legend_count = 0
    
        ind1 = iSV[start_tog][iO2_n] +i*cl['nxt_y']
        ind2 = iSV[start_tog][iO2_n] +i*cl['nxt_y'] +(Nr_cl +shl_tog)*cl['nxt_r']
        species_k_Naf = sol.y[ind1:ind2:n_r_species, :]

        for j in range(Nr_cl +shl_tog):
            plt.figure(fig_num)
            plt.plot(sol.t, species_k_Naf[j])

            legend_str.append(naf_b_ca.species_names[iO2_n] + 'r' + str(legend_count))
            legend_count = legend_count +1

        plt.title('y-node = ' + str(i))
        plt.legend(legend_str)
        plt.ylabel(r'Nafion Phase Mass Densities [kg/m$^3$]')
        plt.xlabel('Time, t [s]')
        plt.tight_layout()
        
        if save:
            plt.savefig('Nafion_Densities_y' + str(i) + '_v_Time.png')
            
        fig_num = fig_num +1

    # Extract gas species densities and plot:
    for i in range(gas_ca.n_species):
        species_gas_cl = sol.y[iSV['rho_gas_k'][i]:L_sv:cl['nxt_y']][0]
        species_gas_gdl = sol.y[iSV['rho_gdl_k'][i]:L_gdl:gdl['nxt_y']][0]

        plt.figure(fig_num)
        plt.plot(sol.t, species_gas_cl)
        
        plt.figure(fig_num+1)
        plt.plot(sol.t, species_gas_gdl)
        
    plt.figure(fig_num)
    plt.legend(gas_ca.species_names)
    plt.ylabel(r'CL Gas Phase Mass Densities [kg/m$^3$]')
    plt.xlabel('Time, t [s]')
    plt.tight_layout()
    
    plt.figure(fig_num+1)
    plt.legend(gas_ca.species_names)
    plt.ylabel(r'GDL Gas Phase Mass Densities [kg/m$^3$]')
    plt.xlabel('Time, t [s]')
    plt.tight_layout()
    
    if save:
        plt.figure(fig_num)
        plt.savefig('CL_Gas_Densities_v_Time.png')
        
        plt.figure(fig_num+1)
        plt.savefig('GDL_Gas_Densities_v_Time.png')
    
    fig_num = fig_num +2
    
    for i in range(Ny_cl):
        # Extract pt surface sites and plot:
        legend = []
        legend_count = 0
        
        for j in range(pt_s_ca.n_species):
            theta_pt_k = sol.y[iSV['theta_pt_k'][j] +i*cl['nxt_y'], :]
            
            plt.figure(fig_num)
            plt.plot(sol.t, theta_pt_k)
            
        fig_num = fig_num +1
        plt.title('y-node = ' + str(i))
        plt.legend(pt_s_ca.species_names)
        plt.ylabel('Surface Coverage [-]')
        plt.xlabel('Time, t [s]')
        plt.tight_layout()
    
    if save == 1:
        plt.savefig('Theta_pt_k_v_Time.png')
        
    fig_num = fig_num +1

if radial:
    # If turned 'on' in pemfc_runner.py, this flag will produce a plot that shows
    # the O2 density in the Nafion phase against the radius of either the shells
    # in the core-shell model or the entire agglomerate for the flooded-agglomerate
    # model. A line is included and commented out below if desired to see the 
    # proton densities, which should be constant due to the Nafion structure.
    
    # Extract O2 density from each Nafion shell as f(r):
    legend_str = []
    legend_count = 0
    
    for i in range(Ny_cl):
        ind_1 = int(iSV[start_tog][iO2_n] +i*cl['nxt_y'])
        ind_2 = int(iSV[start_tog][iO2_n] +i*cl['nxt_y'] +(Nr_cl +shl_tog)*cl['nxt_r'])
        naf_O2_r = sol.y[ind_1:ind_2:cl['nxt_r'], -1]
        
        ind_1 = int(iSV[start_tog][iH_n] +i*cl['nxt_y'])
        ind_2 = int(iSV[start_tog][iH_n] +i*cl['nxt_y'] +(Nr_cl +shl_tog)*cl['nxt_r'])
        naf_H_r = sol.y[ind_1:ind_2:cl['nxt_r'], -1]

        plt.figure(fig_num)
        plt.plot(1/cl['1/r_j']*1e9, naf_O2_r, '-o')
        #plt.plot(1/cl['1/r_j']*1e9, naf_H_r, '-o')
        legend_str.append('y node = ' + str(legend_count))
        legend_count = legend_count +1

    plt.xlabel('Nafion Shell Radius [nm]')
    plt.ylabel('Nafion Phase O2 Density [kg/m$^3$]')
    plt.legend(legend_str, loc='best')
    plt.tight_layout()

    if save:
        plt.savefig('Nafion_O2_v_Radius.png')
        
    fig_num = fig_num +1

if grads:
    # If turned 'on' in pemfc_runner.py, this flag will produce plots of the
    # electrolyte potential, O2 density in the outermost Nafion shells, and the
    # fraction of faradaic current against the CL depth. This helps to ensure 
    # that the gradients are in the correct direction as well as providing insight
    # into where potential and transport losses take place. For the O2 density 
    # in the gas phase, the gradient produced is plotted agains the entire cathode
    # depth (including the GDL and CL).
    
    # Note: since the fraction of faradaic current is not stored in the solution
    # vector, the cantera objects are used to have their states set from the saved
    # information and then i_far is recalculated from the production rates given
    # from the set conditions.
    
    # Make sure potential gradient is in correct direction:
    plt.figure(fig_num)
    Phi_elyte_y = -1*sol.y[iSV['phi_dl']:L_sv:cl['nxt_y'], -1]
    plt.plot(np.linspace(0,t_cl*1e6,Ny_cl), Phi_elyte_y, '-o')

    plt.xlabel(r'Cathode CL Depth [$\mu$m]')
    plt.ylabel('Electrolyte Potential [V]')
    plt.tight_layout()

    if save:
        plt.savefig('Nafion_Potential_v_CL_Depth.png')
        
    fig_num = fig_num +1

    # Make sure O2 gradient is in correct direction - gas phase:
    plt.figure(fig_num)
    iO2 = gas_ca.species_index('O2')
    
    rho_O2_gdl = sol.y[iSV['rho_gdl_k'][iO2_g]:L_gdl:gdl['nxt_y'], -1]
    rho_O2_CL = sol.y[iSV['rho_gas_k'][iO2_g]:L_sv:cl['nxt_y'], -1]
    rho_O2_y = np.hstack([rho_O2_gdl, rho_O2_CL])
    
    dist_ca = np.hstack([np.linspace(0.5*gdl['dy'], t_gdl-0.5*gdl['dy'],Ny_gdl),
                         np.linspace(t_gdl+0.5*cl['dy'], t_gdl+t_cl-0.5*cl['dy'],Ny_cl)])
        
    plt.plot(dist_ca*1e6, rho_O2_y, '-o')

    plt.xlabel(r'Cathode Depth [$\mu$m]')
    plt.ylabel('Gas Phase O2 Density [kg/m$^3$]')
    plt.tight_layout()

    if save:
        plt.savefig('Gas_O2_v_Depth.png')
        
    fig_num = fig_num +1

    # Check O2 gradient direction - naf phase (outermost shell):
    plt.figure(fig_num)
    rho_O2_y = sol.y[int(iSV[start_tog][iO2_n]):L_sv:cl['nxt_y'], -1]
    plt.plot(np.linspace(0,t_cl*1e6,Ny_cl), rho_O2_y, '-o')

    plt.xlabel(r'Cathode CL Depth [$\mu$m]')
    plt.ylabel('Nafion (outermost radius) O2 Density [kg/m$^3$]')
    plt.tight_layout()

    if save:
        plt.savefig('Nafion_O2_v_CL_Depth.png')
        
    fig_num = fig_num +1
    
    # Check i_far as a f(y) in CL at steady-state conditions:
    i_ind = np.argmin(abs(i_ext - i_find))
    sv = sv_save[1:, i_ind]

    y_cl = np.linspace(0, t_cl, cl['Ny']) *1e6
    cl_ymv = 0
    
    if model == 'core_shell':
        i_far_cl = np.zeros(cl['Ny'])
        
        for i in range(cl['Ny']):
            carb_ca.electric_potential = 0
            pt_s_ca.electric_potential = 0
            
            naf_b_ca.electric_potential = -sv[iSV['phi_dl'] +cl_ymv]
            naf_s_ca.electric_potential = -sv[iSV['phi_dl'] +cl_ymv]
            
            rho_naf_k = sv[iSV['rho_naf_k'] +cl_ymv]
            naf_b_ca.TDY = sv[iSV['T_cl'] +cl_ymv], sum(rho_naf_k), rho_naf_k
            pt_s_ca.coverages = sv[iSV['theta_pt_k'] +cl_ymv]
            
            i_far_cl[i] = pt_s_ca.get_net_production_rates(carb_ca)\
                        *ct.faraday *cl['SApv_pt']
            
            cl_ymv = cl_ymv +cl['nxt_y']
                
    elif model == 'flooded_agg':
        i_far_r = np.zeros([cl['Ny'], cl['Nr']])
        
        for i in range(cl['Ny']):
            carb_ca.electric_potential = 0
            pt_s_ca.electric_potential = 0
            
            naf_b_ca.electric_potential = -sv[iSV['phi_dl'] +cl_ymv]
            naf_s_ca.electric_potential = -sv[iSV['phi_dl'] +cl_ymv]
            
            for j in range(cl['Nr']):
                rho_naf_k = sv[iSV['rho_naf_k'] +cl_ymv +j*cl['nxt_r']]
                naf_b_ca.TDY = sv[iSV['T_cl'] +cl_ymv], sum(rho_naf_k), rho_naf_k
                pt_s_ca.coverages = sv[iSV['theta_pt_k'] +cl_ymv +j*cl['nxt_r']]
                
                i_far_r[i,j] = pt_s_ca.get_net_production_rates(carb_ca)\
                              *ct.faraday *cl['SApv_pt'] *cl['Vf_ishl'][j]             
                
            i_far_cl = np.sum(i_far_r, axis=1)
            
            cl_ymv = cl_ymv +cl['nxt_y']
    
    plt.figure(fig_num)
    plt.plot(y_cl, -i_far_cl / (i_ext[i_ind] *cl['1/dy'] *100**2), '-o')
    
    plt.xlabel(r'Cathode Depth [$/mu$m]')
    plt.ylabel(r'i$_{Far}$ / i$_{ext}$ [-]')
    plt.tight_layout()
        
    if save:
        plt.savefit('i_far_v_CL_Depth.png')
        
    fig_num = fig_num +1

if over_p:
    # If turned 'on' in pemfc_runner.py, this flag will produce an overpotential
    # plot at steady-state conditions using all of the external currents specified
    # by the user in the pemfc_runner.py file.
    
    # Plot a overpotential curve (i_ext vs eta_ss) for the cathode:
    plt.figure(fig_num)
    plt.plot(i_ext, eta_ss)

    plt.ylabel(r'Steady-state Overpotential [V]')
    plt.xlabel(r'Current Density [A/cm$^2$]')
    plt.tight_layout()

    if save:
        plt.savefig('Overpotential_Curve.png')
    
    fig_num = fig_num +1

if polar:
    # If turned 'on' in pemfc_runner.py, this flag will produce a polarization
    # plot at steady-state conditions using all of the external currents specified
    # by the user in the pemfc_runner.py file.
    
    # Note: 'data' is an option in pemfc_runner.py. If 'data' is turned 'on' then
    # data from Owejan et. al. will be included on the polarization curve so long
    # as the user was running at a Pt loading of 0.2, 0.1, 0.05, or 0.025 mg/cm^2.
    
    # Plot a polarization curve (i_ext vs dphi_ss) for the cell:
    plt.figure(fig_num)
    plt.plot(i_ext, dphi_ss, label='model', linewidth=2, linestyle='-')

    plt.ylabel(r'Cell Voltage [V]', fontname=font_nm)
    plt.xlabel(r'Current Density [A/cm$^2$]', fontname=font_nm)
    plt.tight_layout()
    
    if all([data, w_Pt == 0.2]):
        x = np.array([0.000, 0.050, 0.200, 0.400, 0.800, 1.000, 1.200, 1.500, 
                      1.650, 1.850, 2.000])
        y = np.array([0.952, 0.849, 0.803, 0.772, 0.731, 0.716, 0.700, 0.675, 
                      0.665, 0.647, 0.634])
        yerr = np.array([0, 0.012, 0.007, 0.007, 0.012, 0.001, 0.008, 0.007,
                         0.007, 0.009, 0.009])
        color = 'C0'
    elif all([data, w_Pt == 0.1}):
        x = np.array([0.000, 0.050, 0.200, 0.400, 0.800, 1.000, 1.200, 1.500, 
                      1.650, 1.850, 2.000])
        y = np.array([0.930, 0.834, 0.785, 0.754, 0.711, 0.691, 0.673, 0.649, 
                      0.635, 0.615, 0.598])
        yerr = np.array([0, 0.009, 0.007, 0.005, 0.007, 0.011, 0.011, 0.007, 
                         0.009, 0.011, 0.011])
        color = 'C1'
    elif all([data, w_Pt == 0.05]):
        x = np.array([0.000, 0.050, 0.200, 0.400, 0.800, 1.000, 1.200, 1.500, 
                      1.650, 1.850, 2.000])
        y = np.array([0.919, 0.810, 0.760, 0.724, 0.674, 0.653, 0.634, 0.603,
                      0.585, 0.558, 0.537])
        yerr = np.array([0, 0.008, 0.006, 0.006, 0.007, 0.007, 0.005, 0.005, 
                         0.006, 0.007, 0.007])
        color = 'C2'
    elif all([data, w_Pt == 0.025]):
        x = np.array([0.000, 0.050, 0.200, 0.400, 0.800, 1.000, 1.200, 1.500, 
                      1.650, 1.850, 2.000])
        y = np.array([0.910, 0.785, 0.724, 0.683, 0.626, 0.598, 0.572, 0.527, 
                      0.502, 0.463, 0.430])
        yerr = np.array([0, 0.004, 0.010, 0.014, 0.013, 0.013, 0.019, 0.024, 
                         0.025, 0.023, 0.024])
        color = 'C3'
        
    try:
        plt.errorbar(x, y, yerr=yerr, fmt='o', color=color, capsize=3, label='Owejan et. al.')
        plt.ylim([0.35, 1.0]); plt.xlim([0, 2.1]) #; plt.legend(loc='best')
    except:
        None

    if save:
        plt.savefig('Polarization_Curve.png')
        
    fig_num = fig_num +1
    
if i_ver:
    # If turned 'on' in pemfc_runner.py, this flag will calculate the O2 flux
    # at the GDL and CL boundary based on the value of 'i_find' also set in the
    # pemfc_runner.py file. Using stoichiometry and theory, the flux should be
    # i_ext / 4*F. This value is compared to the flux found using Fick's law from
    # the model's stored gas-phase densities. Making sure these values match
    # provides additional verification that the model is functioning properly
    # and that boundary conditions are appropriately being met.
    
    i_ind = np.argmin(abs(i_ext - i_find))
    sv = sv_save[1:, i_ind]

    i_4F = i_ext[i_ind]*100**2 / (4*ct.faraday)
    print('\nO2_i_4F:', i_4F)

    i_Last_gdl = int(L_gdl /Ny_gdl *(Ny_gdl -1))
    rho_gdl_k = sv[iSV['rho_gdl_k'] +i_Last_gdl]
    TDY1 = sv[iSV['T_gdl'] +i_Last_gdl], sum(rho_gdl_k), rho_gdl_k

    i_First_cl = 0
    rho_cl_k = sv[iSV['rho_gas_k'] +i_First_cl]
    TDY2 = sv[iSV['T_cl'] +i_First_cl], sum(rho_cl_k), rho_cl_k

    O2_BC_flux = fickian_adf(TDY1, TDY2, gas_ca, gdl_cl, 1) \
              / gas_ca.molecular_weights

    print('i_ext:', i_ext[i_ind], 'O2_BC_flux:', O2_BC_flux[0],
          'ratio:', i_4F / O2_BC_flux[0])
    
# Move back to original cwd after files are saved:
os.chdir(cwd)

plt.show()