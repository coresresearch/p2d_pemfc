""" Import needed modules """
"-----------------------------------------------------------------------------"
import cantera as ct
import numpy as np

""" Define ODE for Solution """
"-----------------------------------------------------------------------------"
def dSVdt_func(t, SV, objs, params, ptrs):
    # Extract dictionaries for readability:------------------------------------
    metal_ca = objs['metal_ca']
    gas_ca = objs['gas_ca']
    naf_bulk_ca = objs['naf_bulk_ca']
    Pt_surf_ca = objs['Pt_surf_ca']
    naf_surf_ca = objs['naf_surf_ca']
    
    SVptr = ptrs['SVptr']
    Naf_ptr = ptrs['Nafptr']
    Pt_ptr = ptrs['Ptptr']

    # Reset dSVdt for next iteration:
    dSVdt = np.zeros_like(SV)

    " Tracking variables for SV spacing - done for looping "
    SV_shift = 0 # Start with the first node (carbon -> gas, top -> bottom)
    SV_spacing = int(len(params['SV_0'])*params['Ny_inv'])
    
    """ Boundary Condition - CC side with O2 flow """   
    # Ionic currents at BC (top):----------------------------------------------
    i_io_up = 0
    
    # State gas conditions of current node (top):------------------------------
    rho_gas_array = SV[SVptr['rho_gas_k']]
    TDY_1 = SV[SVptr['temp']], sum(rho_gas_array), rho_gas_array 
                 
    # Density of gas species at BC (top):--------------------------------------
    gas_ca.TPY = params['TPY_gas_ca_BC']
    D_eff_BC = gas_ca.mix_diff_coeffs_mass
    rho_BC = gas_ca.density_mass
    Y_BC = gas_ca.Y
    
    gas_ca.TDY = TDY_1
    D_eff_1 = gas_ca.mix_diff_coeffs_mass
    rho_1 = gas_ca.density_mass
    Y_1 = gas_ca.Y
    
    D_eff_gas = np.mean([D_eff_BC, D_eff_1])
    rho_av = np.mean([rho_BC, rho_1])
    
    flux_gas_up = params['phi/tau_sq']*D_eff_gas*rho_av\
                * (Y_BC - Y_1)*params['1/dy']
    
    # Density of naf species initialization:-----------------------------------
    naf_shift = naf_bulk_ca.n_species-1

    """ Generic loop for interal nodes in y-direction """
    for i in range(params['Ny']-1):
        # Set the inner shell state:-------------------------------------------
        phi_dl = SV[SVptr['phi_dl']+SV_shift]
        metal_ca.electric_potential = 0
        Pt_surf_ca.electric_potential = 0

        naf_bulk_ca.electric_potential = -phi_dl
        naf_surf_ca.electric_potential = -phi_dl

        rho_naf_array = np.hstack((params['H_density'],
                                   SV[SVptr['rho_n1']+SV_shift]))

        T_ca = SV[SVptr['temp']+SV_shift] 
        naf_bulk_ca.TDY = T_ca, sum(rho_naf_array), rho_naf_array

        # Double layer voltage at cathode:-------------------------------------
        i_io_down = params['sig_naf_io']\
                  * (SV[SVptr['phi_dl']+SV_shift] 
                  - SV[SVptr['phi_dl']+SV_shift+SV_spacing])*params['1/dy']
        
        sdot_Pts_e = Pt_surf_ca.net_production_rates[params['index_electron']]
        i_Far = sdot_Pts_e*ct.faraday
        
        i_dl = (i_io_up - i_io_down)*params['1/dy']\
             - i_Far*params['A_pv_surf_Pt']

        dSVdt[SVptr['phi_dl']+SV_shift] = i_dl*params['(C*A)_inv']
        
        i_io_up = i_io_down

        # Temperature of each node in Y:---------------------------------------
        dSVdt[SVptr['temp']+SV_shift] = 0
        
        # Inner shell reactions:-----------------------------------------------
        sdot_Pts_Naf = Pt_surf_ca.net_production_rates[Pt_ptr['Naf']]\
                     * naf_bulk_ca.molecular_weights[Naf_ptr['Naf']]
        
        # Outer shell reactions:-----------------------------------------------
        rho_naf_array = np.hstack((params['H_density'], 
                    SV[SVptr['rho_n1']+(params['Nr']-1)*naf_shift+SV_shift]))
        
        naf_bulk_ca.TDY = T_ca, sum(rho_naf_array), rho_naf_array
        
        sdot_nafs_gas = naf_surf_ca.net_production_rates[Naf_ptr['gas']]\
                      * gas_ca.molecular_weights
                      
        sdot_nafs_Naf = naf_surf_ca.net_production_rates[Naf_ptr['Naf']]\
                      * naf_bulk_ca.molecular_weights[Naf_ptr['Naf']]
        
        # Density of gas species:----------------------------------------------                        
        rho_gas_array = SV[SVptr['rho_gas_k']+SV_shift+SV_spacing]
        gas_ca.TDY = T_ca, sum(rho_gas_array), rho_gas_array 
                     
        D_eff_2 = gas_ca.mix_diff_coeffs_mass
        rho_2 = gas_ca.density_mass
        Y_2 = gas_ca.Y
    
        D_eff_gas = np.mean([D_eff_2, D_eff_1])
        rho_av = np.mean([rho_1, rho_2])
    
        flux_gas_down = params['phi/tau_sq']*D_eff_gas*rho_av\
                      * (Y_1 - Y_2)*params['1/dy']
        
#        dSVdt[SVptr['rho_gas_k']+SV_shift] = params['eps_gas_inv']\
#                  * params['A_pv_naf_gas_int']*sdot_nafs_gas\
#                  + (flux_gas_up - flux_gas_down)*params['1/dy']
                  
        flux_gas_up = flux_gas_down
        D_eff_1 = D_eff_2
        Y_1 = Y_2

        # Density of nafion species:-------------------------------------------
        "Innermost node has Pt surface reaction BC and diffusion in"       
        flux_in = params['D_eff_naf']*(params['1/r_nodes'][0])**2\
                * (((params['r_bounds'][0])**2\
                * (SV[SVptr['rho_n1']+naf_shift+SV_shift]
                - SV[SVptr['rho_n1']+SV_shift])\
                * params['1/delta_rs'][0])* params['1/t_shells'][0])

        dSVdt[SVptr['rho_n1']+SV_shift] = \
                  params['eps_naf_inv']*sdot_Pts_Naf*params['A_pv_surf_Pt']\
                + flux_in

        "Center nodes have radial diffusion"
        flux_out = flux_in

        for j in range(1, params['Nr']-1):
            flux_in = params['D_eff_naf']*(params['1/r_nodes'][j])**2\
                    * ((params['r_bounds'][j])**2\
                    * (SV[SVptr['rho_n1']+(j+1)*naf_shift+SV_shift]
                    - SV[SVptr['rho_n1']+j*naf_shift+SV_shift])\
                    * params['1/delta_rs'][j])*params['1/t_shells'][j]

            dSVdt[SVptr['rho_n1']+j*naf_shift+SV_shift] = flux_in - flux_out

            flux_out = flux_in

        "Outermost node has gas-interface reaction BC and diffusion out"                      
        dSVdt[SVptr['rho_n1']+(j+1)*naf_shift+SV_shift] = \
               params['eps_naf_inv']*sdot_nafs_Naf*params['A_pv_naf_gas_int']\
             - flux_out
             
        SV_shift = SV_shift + SV_spacing
    
    """ Boundary Condition - Elyte side """
    # Set the inner shell state:-----------------------------------------------
    phi_dl = SV[SVptr['phi_dl']+SV_shift]
    metal_ca.electric_potential = 0
    Pt_surf_ca.electric_potential = 0

    naf_bulk_ca.electric_potential = -phi_dl
    naf_surf_ca.electric_potential = -phi_dl

    rho_naf_array = np.hstack((params['H_density'], 
                               SV[SVptr['rho_n1']+SV_shift]))

    T_ca = SV[SVptr['temp']+SV_shift]
    naf_bulk_ca.TDY = T_ca, sum(rho_naf_array), rho_naf_array
    
    # Double layer voltage at BC (bottom):-------------------------------------
    i_io_down = params['i_ext']*params['t_ca_inv']
    
    sdot_Pts_e = Pt_surf_ca.net_production_rates[params['index_electron']]
    i_Far = sdot_Pts_e*ct.faraday
    
    i_dl = i_io_up*params['1/dy'] - i_io_down - i_Far*params['A_pv_surf_Pt']

    dSVdt[SVptr['phi_dl']+SV_shift] = i_dl*params['(C*A)_inv']
        
    # Temperature at BC (bottom):----------------------------------------------
    dSVdt[SVptr['temp']+SV_shift] = 0
    
    # Inner shell reactions at BC (bottom):------------------------------------
    sdot_Pts_Naf = Pt_surf_ca.net_production_rates[Pt_ptr['Naf']]\
                 * naf_bulk_ca.molecular_weights[Naf_ptr['Naf']]
    
    # Outer shell reactions at BC (bottom):------------------------------------
    rho_naf_array = np.hstack((params['H_density'], 
                    SV[SVptr['rho_n1']+(params['Nr']-1)*naf_shift+SV_shift]))
        
    naf_bulk_ca.TDY = T_ca, sum(rho_naf_array), rho_naf_array
    
    sdot_nafs_gas = naf_surf_ca.net_production_rates[Naf_ptr['gas']]\
                  * gas_ca.molecular_weights
                  
    sdot_nafs_Naf = naf_surf_ca.net_production_rates[Naf_ptr['Naf']]\
                  * naf_bulk_ca.molecular_weights[Naf_ptr['Naf']]
    
    # Density of gas species at BC (bottom):-----------------------------------                               
    flux_gas_down = np.zeros(gas_ca.n_species)
        
#    dSVdt[SVptr['rho_gas_k']+SV_shift] = params['eps_gas_inv']\
#                  * params['A_pv_naf_gas_int']*sdot_nafs_gas\
#                  + (flux_gas_up - flux_gas_down)*params['1/dy']
    
    # Density of nafion species (bottom shell):--------------------------------
    "Innermost node has Pt surface reaction BC and diffusion in"
    flux_in = params['D_eff_naf']*(params['1/r_nodes'][0])**2\
            * (((params['r_bounds'][0])**2\
            * (SV[SVptr['rho_n1']+naf_shift+SV_shift]
            - SV[SVptr['rho_n1']+SV_shift])\
            * params['1/delta_rs'][0])* params['1/t_shells'][0])

    dSVdt[SVptr['rho_n1']+SV_shift] = \
              params['eps_naf_inv']*sdot_Pts_Naf*params['A_pv_surf_Pt']\
            + flux_in

    "Center nodes have radial diffusion"
    flux_out = flux_in

    for j in range(1, params['Nr']-1):
        flux_in = params['D_eff_naf']*(params['1/r_nodes'][j])**2\
                * ((params['r_bounds'][j])**2\
                * (SV[SVptr['rho_n1']+(j+1)*naf_shift+SV_shift]
                - SV[SVptr['rho_n1']+j*naf_shift+SV_shift])\
                * params['1/delta_rs'][j])*params['1/t_shells'][j]

        dSVdt[SVptr['rho_n1']+j*naf_shift+SV_shift] = flux_in - flux_out

        flux_out = flux_in

    "Outermost node has gas-interface reaction BC and diffusion out"
    dSVdt[SVptr['rho_n1']+(j+1)*naf_shift+SV_shift] = \
           params['eps_naf_inv']*sdot_nafs_Naf*params['A_pv_naf_gas_int']\
         - flux_out
         
#    print('Phi:', dSVdt[SVptr['phi_dl']:len(SV):int(len(SV)*params['Ny_inv'])])
#    print('temp:', dSVdt[SVptr['temp']:len(SV):int(len(SV)*params['Ny_inv'])])
#    for k in range(params['Nr']):
#        print('Rho Naf:', dSVdt[SVptr['rho_naf_k'][k]:len(SV):int(len(SV)*params['Ny_inv'])])
#    for k in range(gas_ca.n_species):
#        print('Rho Gas '+str(gas_ca.species_names[k])+':', dSVdt[SVptr['rho_gas_k'][k]:len(SV):int(len(SV)*params['Ny_inv'])])
#    input('press enter')
         
    return dSVdt
