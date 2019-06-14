""" Import needed modules """
"-----------------------------------------------------------------------------"
import pylab as plt
import numpy as np



""" Post-processing for Plotting and Other Results """
"-----------------------------------------------------------------------------"
# Set font size:
font_nm, font_sz = 'Arial', 14
font = plt.matplotlib.font_manager.FontProperties(family=font_nm, size=font_sz)
plt.rcParams.update({'font.size': font_sz})

# Move into correct directory to save any outputs:
if save == 1:
    os.chdir(cwd + '/' + folder_name)
    
# Generalize figure numbering:
fig_num = 1

if debug == 1:
    # Extract cathode double layer potential and plot:
    phi_dl = sol.y[iSV['phi_dl'], :]

    plt.figure(fig_num)
    plt.plot(sol.t, phi_dl)

    plt.ylabel(r'Cathode Double Layer Voltage, $\phi_{dl}$ [V]')
    plt.xlabel('Time, t [s]')
    plt.tight_layout()
    
    if save == 1:
        plt.savefig('Double_Layer_v_Time.png')
        
    fig_num = fig_num +1

    # Extract nafion species densities and plot:
    legend_str = []
    legend_count = 0

    for i in range(Ny):
        species_k_Naf = sol.y[iSV['rho_naf_k'] + i*int(L_cl/Ny), :]

        for j in range(Nr):
            plt.figure(fig_num)
            plt.plot(sol.t, species_k_Naf[j])

            legend_str.append(naf_b_ca.species_names[1] + str(legend_count))
            legend_count = legend_count +1

        plt.title('y-node = ' + str(i))
        plt.legend(legend_str)
        plt.ylabel(r'Nafion Phase Mass Densities, $\rho$ [kg/m$^3$]')
        plt.xlabel('Time, t [s]')
        plt.tight_layout()
        
        if save == 1:
            plt.savefig('Nafion_Densities_y' + str(i) + '_v_Time.png')
            
        fig_num = fig_num +1

    # Extract gas species densities and plot:
    for i in range(gas_ca.n_species):
        species_gas_cl = sol.y[iSV['rho_gas_k'][i]::L_cl][0]
        species_gas_gdl = sol.y[iSV['rho_gdl_k'][i]::L_cl][0]

        plt.figure(fig_num)
        plt.plot(sol.t, species_gas_cl)
        
        plt.figure(fig_num+1)
        plt.plot(sol.t, species_gas_gdl)
        
    plt.figure(fig_num)
    plt.legend(gas_ca.species_names)
    plt.ylabel(r'CL Gas Phase Mass Densities, $\rho$ [kg/m$^3$]')
    plt.xlabel('Time, t [s]')
    plt.tight_layout()
    
    plt.figure(fig_num+1)
    plt.legend(gas_ca.species_names)
    plt.ylabel(r'GDL Gas Phase Mass Densities, $\rho$ [kg/m$^3$]')
    plt.xlabel('Time, t [s]')
    plt.tight_layout()
    
    if save == 1:
        plt.figure(fig_num)
        plt.savefig('CL_Gas_Densities_v_Time.png')
        
        plt.figure(fig_num+1)
        plt.savefig('GDL_Gas_Densities_v_Time.png')
    
    fig_num = fig_num +2

if radial == 1:
    # Extract O2 density from each Nafion shell as f(r):
    legend_str = []
    legend_count = 0

    for i in range(Ny):
        ind_1 = int(iSV['rho_n1'] + i*L_cl/Ny)
        ind_2 = int(iSV['rho_n1'] + i*L_cl/Ny + Nr)
        naf_k_r = sol.y[ind_1:ind_2, -1]

        plt.figure(fig_num)
        plt.plot(1/p['1/r_j']*1e9, naf_k_r*1e4)
        legend_str.append('y node = ' + str(legend_count))
        legend_count = legend_count +1

    plt.xlabel('Nafion Shell Radius, [nm]')
    plt.ylabel('Nafion Phase - O2 Density *1e4 [$kg/m^3$]')
    plt.legend(legend_str, loc='lower right')
    plt.tight_layout()

    if save == 1:
        plt.savefig('Nafion_O2_v_Radius.png')
        
    fig_num = fig_num +1

if grads == 1:  
    # Make sure potential gradient is in correct direction:
    plt.figure(fig_num)
    Phi_elyte_y = -1*sol.y[iSV['phi_dl']:L_cl:int(L_cl/Ny), -1]
    plt.plot(np.linspace(0,t_cl*1e6,Ny), Phi_elyte_y, '-o')

    plt.xlabel(r'Cathode CL Depth [$\mu$m]')
    plt.ylabel('Electrolyte Potential [V]')
    plt.tight_layout()

    if save == 1:
        plt.savefig('Nafion_Potential_v_CL_Depth.png')
        
    fig_num = fig_num +1

    # Make sure O2 gradient is in correct direction - gas phase:
    plt.figure(fig_num)
    iO2 = gas_ca.species_index('O2')
    
    rho_O2_gdl = sol.y[iSV['rho_gdl_k'][iO2]::int(L_gdl/Ny_gdl), -1]
    rho_O2_CL = sol.y[iSV['rho_gas_k'][iO2]:L_cl:int(L_cl/Ny), -1]
    rho_O2_y = np.hstack([rho_O2_gdl, rho_O2_CL])
    
    dist_ca = np.hstack([np.linspace(0.5*p['gdl']['dy'], 
                         t_gdl-0.5*p['gdl']['dy'],Ny_gdl),
                         np.linspace(t_gdl+0.5*p['dy'], 
                         t_gdl+t_cl-0.5*p['dy'],Ny)])
        
    plt.plot(dist_ca*1e6, rho_O2_y, '-o')

    plt.xlabel(r'Cathode Depth [$\mu$m]')
    plt.ylabel('Gas Phase - rho_O2 [$kg/m^3$]')
    plt.tight_layout()

    if save == 1:
        plt.savefig('Gas_O2_v_Depth.png')
        
    fig_num = fig_num +1

    # Check O2 gradient direction - naf phase (inner shell):
    plt.figure(fig_num)
    rho_O2_y = sol.y[int(iSV['rho_n1']):L_cl:int(L_cl/Ny), -1]
    plt.plot(np.linspace(0,t_cl*1e6,Ny), rho_O2_y*1e4, '-o')

    plt.xlabel(r'Cathode CL Depth [$\mu$m]')
    plt.ylabel('Nafion (inner shell) - rho_O2 *1e4 [$kg/m^3$]')
    plt.tight_layout()

    if save == 1:
        plt.savefig('Nafion_O2_v_CL_Depth.png')
        
    fig_num = fig_num +1

if over_p == 1:
    # Plot a overpotential curve (i_ext vs eta_ss) for the cathode:
    plt.figure(fig_num)
    plt.plot(np.hstack([i_OCV, i_ext]), eta_ss)

    plt.ylabel(r'Steady-state Overpotential, $\eta_{ss}$ [V]')
    plt.xlabel(r'External Current, $i_{ext}$ [$A/cm^2$]')
    plt.tight_layout()

    if save == 1:
        plt.savefig('Overpotential_Curve.png')
    
    fig_num = fig_num +1

if polar == 1:
    # Plot a polarization curve (i_ext vs dphi_ss) for the cell:
    plt.figure(fig_num)
    plt.plot(np.hstack([i_OCV, i_ext]), dphi_ss, label='model', linewidth=2, linestyle='--')

    plt.ylabel(r'Cell Voltage [V]', fontname=font_nm)
    plt.xlabel(r'Current Density [A/cm$^2$]', fontname=font_nm)
    plt.tight_layout()
    
    if w_Pt == 0.2:
        x = np.array([0.008, 0.051, 0.201, 0.403, 0.802, 1.002, 1.202, 1.501, 
                      1.651, 1.851, 2.000])
        y = np.array([0.952, 0.849, 0.803, 0.772, 0.731, 0.716, 0.700, 0.675, 
                      0.665, 0.647, 0.634])
        yerr = np.array([0, 0.012, 0.007, 0.007, 0.012, 0.001, 0.008, 0.007,
                         0.007, 0.009, 0.009])
        color = 'C0'
    elif w_Pt == 0.1:
        x = np.array([0.006, 0.053, 0.201, 0.401, 0.802, 1.002, 1.200, 1.499, 
                      1.651, 1.851, 2.000])
        y = np.array([0.930, 0.834, 0.785, 0.754, 0.711, 0.691, 0.673, 0.649, 
                      0.635, 0.615, 0.598])
        yerr = np.array([0, 0.009, 0.007, 0.005, 0.007, 0.011, 0.011, 0.007, 
                         0.009, 0.011, 0.011])
        color = 'C1'
    elif w_Pt == 0.05:
        x = np.array([0.008, 0.053, 0.201, 0.401, 0.800, 1.000, 1.200, 1.500, 
                      1.651, 1.850, 2.001])
        y = np.array([0.919, 0.810, 0.760, 0.724, 0.674, 0.653, 0.634, 0.603,
                      0.585, 0.558, 0.537])
        yerr = np.array([0, 0.008, 0.006, 0.006, 0.007, 0.007, 0.005, 0.005, 
                         0.006, 0.007, 0.007])
        color = 'C2'
    elif w_Pt == 0.025:
        x = np.array([0.003, 0.049, 0.202, 0.404, 0.803, 1.005, 1.204, 1.503, 
                      1.653, 1.851, 2.004])
        y = np.array([0.910, 0.785, 0.724, 0.683, 0.626, 0.598, 0.572, 0.527, 
                      0.502, 0.463, 0.430])
        yerr = np.array([0, 0.004, 0.010, 0.014, 0.013, 0.013, 0.019, 0.024, 
                         0.025, 0.023, 0.024])
        color = 'C3'
        
    try:
        None
#        plt.errorbar(x, y, yerr=yerr, fmt='o', color=color, capsize=3, label='Owejan et. al.')
#        plt.ylim([0.35, 1.0])
#        plt.xlim([0, 2.1])
    except:
        None
        
#    plt.legend(loc='best')

    if save == 1:
        plt.savefig('Polarization_Curve.png')
        
    fig_num = fig_num +1
    
# Move back to original cwd after files are saved:
os.chdir(cwd)

plt.show()