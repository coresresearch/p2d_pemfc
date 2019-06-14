""" Import needed modules """
"-----------------------------------------------------------------------------"
from scipy.integrate import solve_ivp
from pemfc_transport_funcs import *
import cantera as ct
import numpy as np
import sys



""" Define ODE for Solution """
"-----------------------------------------------------------------------------"
def dsvdt_func(t, sv, objs, p, ptrs):
    # Toggles to turn on/off inner/outer rxns and gas transports:--------------
    i_rxn = 1
    o_rxn = 1
    gas_tog = 1
    gdl_tog = 1

    # Extract dictionaries for readability:------------------------------------
    carb_ca = objs['carb_ca']
    gas_ca = objs['gas_ca']
    naf_b_ca = objs['naf_b_ca']
    pt_s_ca = objs['pt_s_ca']
    naf_s_ca = objs['naf_s_ca']

    iSV = ptrs['iSV']
    iNaf = ptrs['iNaf']
    iPt = ptrs['iPt']

    # Reset dsvdt for current iteration:
    dsvdt = np.zeros_like(sv)
    
    iout = p['Nr'] -1 # common index multiplier to find outter-most node

    " Tracking variables for SV spacing - done for looping "
    # nxt is the index difference between adjacent nodes. This spacing
    # is the same regardless of which variable is being examined. The SV terms
    # refer to the CL and the GDL has its own moving (mv) and spacing terms.
    
    sv_mv = 0 # Start with the first node (y: GDL -> Elyte, r: in -> out)
    sv_nxt = int(L_cl*p['1/Ny'])

    gdl_mv = 0 # Start with the first GDL node (y: gas channel -> CL)
    gdl_nxt = gas_ca.n_species + 1

    """ Boundary Condition - CC side with O2 flow """
    # Ionic currents at BC (top) - no protons flow into the GDL:---------------
    i_io_up = 0
    
    """ Bondary Condition - GDL and CL gas transport """
    # Densities/Temp of GDL gas species and CL BC (top):-----------------------
    gas_ca.TPY = p['TPY_gas_ca_BC']
    TDY_BC = gas_ca.TDY
    
    # If GDL diffusion is turned on, compare adjacent nodes with ADF flux to 
    # determine the BC composition between the GDL and CL.
    rho_gas_k = sv[iSV['rho_gdl_k']]
    TDY_1 = sv[iSV['T_gdl']], sum(rho_gas_k), rho_gas_k
    
    flux_g_up = fickian_adf(TDY_BC, TDY_1, gas_ca, p['gdl'], gdl_tog)
    
    for k in range(p['Ny_gdl']-1):
        rho_gas_k = sv[iSV['rho_gdl_k']+gdl_mv+gdl_nxt]
        TDY_2 = sv[iSV['T_gdl']], sum(rho_gas_k), rho_gas_k
        
        flux_g_dwn = fickian_adf(TDY_1, TDY_2, gas_ca, p['gdl'], gdl_tog)
        
        igdl = iSV['rho_gdl_k']+gdl_mv
        dsvdt[igdl] = (flux_g_up - flux_g_dwn)*p['1/eps_gdl']*p['gdl']['1/dy']
                                                                                                                                            
        flux_g_up = flux_g_dwn
        TDY_1 = TDY_2
        
        gdl_mv = gdl_mv + gdl_nxt
        
    # Use the composition and state of the last GDL node to calculate the flux
    # into the first CL node.
    rho_gas_k = sv[iSV['rho_gas_k']]
    TDY_2 = sv[iSV['T']], sum(rho_gas_k), rho_gas_k
    
    flux_g_dwn = fickian_adf(TDY_1, TDY_2, gas_ca, p['gdl_cl'], gdl_tog)
    
    iLast = iSV['rho_gdl_k']+gdl_mv
    dsvdt[iLast] = (flux_g_up - flux_g_dwn)*p['1/eps_gdl']*p['gdl']['1/dy']
    
    flux_g_up = fickian_adf(TDY_1, TDY_2, gas_ca, p['gdl_cl'], gas_tog)
    TDY_1 = TDY_2

    # Move index through Nafion shells:----------------------------------------
    naf_mv = naf_b_ca.n_species -1 # the -1 ignores H+

    """ Generic loop for interal nodes in y-direction """
    for i in range(p['Ny']-1):
        # Set the inner shell state:-------------------------------------------
        phi_dl = sv[iSV['phi_dl']+sv_mv]
        carb_ca.electric_potential = 0
        pt_s_ca.electric_potential = 0

        naf_b_ca.electric_potential = -phi_dl
        naf_s_ca.electric_potential = -phi_dl

        rho_naf_k = np.hstack((p['rho_H'], sv[iSV['rho_n1']+sv_mv]))

        T_ca = sv[iSV['T']+sv_mv]
        naf_b_ca.TDY = T_ca, sum(rho_naf_k), rho_naf_k

        # Double layer voltage at cathode:-------------------------------------
        i_io_dwn = (sv[iSV['phi_dl']+sv_mv] - sv[iSV['phi_dl']+sv_mv+sv_nxt])\
                 * p['sig_naf_io']*p['1/dy']

        sdot_e = i_rxn*pt_s_ca.net_production_rates[p['ind_e']]
        i_Far = sdot_e*ct.faraday

        i_dl = (i_io_up - i_io_dwn)*p['1/dy'] - i_Far*p['SA_pv_pt']
        dsvdt[iSV['phi_dl']+sv_mv] = i_dl*p['1/(C*A)']

        i_io_up = i_io_dwn

        # Temperature of each node in Y:---------------------------------------
        dsvdt[iSV['T']+sv_mv] = 0

        # Inner shell reactions:-----------------------------------------------
        sdot_i_naf = i_rxn*pt_s_ca.net_production_rates[iPt['Naf']]\
                   * naf_b_ca.molecular_weights[iNaf['Naf']]

        # Outer shell reactions:-----------------------------------------------
        gas_ca.TDY = TDY_1

        rho_naf_k = np.hstack((p['rho_H'],sv[iSV['rho_n1']+iout*naf_mv+sv_mv]))
        naf_b_ca.TDY = T_ca, sum(rho_naf_k), rho_naf_k

        sdot_o_gas = o_rxn*naf_s_ca.net_production_rates[iNaf['gas']]\
                   * gas_ca.molecular_weights*gas_tog

        sdot_o_naf = o_rxn*naf_s_ca.net_production_rates[iNaf['Naf']]\
                   * naf_b_ca.molecular_weights[iNaf['Naf']]

        # Density of gas species:----------------------------------------------
        rho_gas_k = sv[iSV['rho_gas_k']+sv_mv+sv_nxt]
        TDY_2 = T_ca, sum(rho_gas_k), rho_gas_k

        flux_g_dwn = fickian_adf(TDY_1, TDY_2, gas_ca, p, gas_tog)

        igas = iSV['rho_gas_k']+sv_mv
        dsvdt[igas] = p['1/eps_gas']*((flux_g_up - flux_g_dwn)*p['1/dy'] 
                                      + p['SA_pv_naf']*sdot_o_gas)

        flux_g_up = flux_g_dwn
        TDY_1 = TDY_2

        # Density of nafion species:-------------------------------------------
        "Innermost node has Pt surface reaction BC and diffusion in"
        rho_k1 = sv[iSV['rho_n1']+sv_mv]
        rho_k2 = sv[iSV['rho_n1']+naf_mv+sv_mv]
        flux_in = radial_fdiff(rho_k1, rho_k2, p, 0)
        
        iFirst = iSV['rho_n1']+sv_mv
        dsvdt[iFirst] = p['1/eps_naf']*sdot_i_naf*p['SA_pv_pt'] + flux_in

        "Center nodes have radial diffusion"
        rho_k1 = rho_k2
        flux_out = flux_in

        for j in range(1, p['Nr']-1):
            rho_k2 = sv[iSV['rho_n1']+(j+1)*naf_mv+sv_mv]
            flux_in = radial_fdiff(rho_k1, rho_k2, p, j)

            dsvdt[iSV['rho_n1']+j*naf_mv+sv_mv] = flux_in - flux_out 
            
            rho_k1 = rho_k2                                          
            flux_out = flux_in

        "Outermost node has gas-interface reaction BC and diffusion out"
        iLast = iSV['rho_n1']+(j+1)*naf_mv+sv_mv
        dsvdt[iLast] = p['1/eps_naf']*sdot_o_naf*p['SA_pv_naf'] - flux_out

        sv_mv = sv_mv + sv_nxt

    """ Boundary Condition - Elyte side """
    # Set the inner shell state:-----------------------------------------------
    phi_dl = sv[iSV['phi_dl']+sv_mv]
    carb_ca.electric_potential = 0
    pt_s_ca.electric_potential = 0

    naf_b_ca.electric_potential = -phi_dl
    naf_s_ca.electric_potential = -phi_dl

    rho_naf_k = np.hstack((p['rho_H'], sv[iSV['rho_n1']+sv_mv]))

    T_ca = sv[iSV['T']+sv_mv]
    naf_b_ca.TDY = T_ca, sum(rho_naf_k), rho_naf_k

    # Double layer voltage at BC (bottom):-------------------------------------
    i_io_dwn = p['i_ext']

    sdot_e = i_rxn*pt_s_ca.net_production_rates[p['ind_e']]
    i_Far = sdot_e*ct.faraday

    i_dl = (i_io_up - i_io_dwn)*p['1/dy'] - i_Far*p['SA_pv_pt']

    dsvdt[iSV['phi_dl']+sv_mv] = i_dl*p['1/(C*A)']

    # Temperature at BC (bottom):----------------------------------------------
    dsvdt[iSV['T']+sv_mv] = 0

    # Inner shell reactions at BC (bottom):------------------------------------
    sdot_i_naf = i_rxn*pt_s_ca.net_production_rates[iPt['Naf']]\
               * naf_b_ca.molecular_weights[iNaf['Naf']]

    # Outer shell reactions at BC (bottom):------------------------------------
    gas_ca.TDY = TDY_1

    rho_naf_k = np.hstack((p['rho_H'], sv[iSV['rho_n1']+iout*naf_mv+sv_mv]))
    naf_b_ca.TDY = T_ca, sum(rho_naf_k), rho_naf_k

    sdot_o_gas = o_rxn*naf_s_ca.net_production_rates[iNaf['gas']]\
               * gas_ca.molecular_weights*gas_tog

    sdot_o_naf = o_rxn*naf_s_ca.net_production_rates[iNaf['Naf']]\
               * naf_b_ca.molecular_weights[iNaf['Naf']]

    # Density of gas species at BC (bottom):-----------------------------------
    flux_g_dwn = np.zeros(gas_ca.n_species)

    igas = iSV['rho_gas_k']+sv_mv
    dsvdt[igas] = p['1/eps_gas']*((flux_g_up - flux_g_dwn)*p['1/dy']
                                  + p['SA_pv_naf']*sdot_o_gas)

    # Density of nafion species (bottom shell):--------------------------------
    "Innermost node has Pt surface reaction BC and diffusion in"
    rho_k1 = sv[iSV['rho_n1']+sv_mv]
    rho_k2 = sv[iSV['rho_n1']+naf_mv+sv_mv]
    flux_in = radial_fdiff(rho_k1, rho_k2, p, 0)
    
    iFirst = iSV['rho_n1']+sv_mv
    dsvdt[iFirst] = p['1/eps_naf']*sdot_i_naf*p['SA_pv_pt'] + flux_in

    "Center nodes have radial diffusion"
    rho_k1 = rho_k2
    flux_out = flux_in

    for j in range(1, p['Nr']-1):
        rho_k2 = sv[iSV['rho_n1']+(j+1)*naf_mv+sv_mv]
        flux_in = radial_fdiff(rho_k1, rho_k2, p, j)

        dsvdt[iSV['rho_n1']+j*naf_mv+sv_mv] = flux_in - flux_out
        
        rho_k1 = rho_k2                                            
        flux_out = flux_in

    "Outermost node has gas-interface reaction BC and diffusion out"
    iLast = iSV['rho_n1']+(j+1)*naf_mv+sv_mv
    dsvdt[iLast] = p['1/eps_naf']*sdot_o_naf*p['SA_pv_naf'] - flux_out

#    print(t)
#    print(sv[iSV['T']:sv_nxt*p['Ny']:sv_nxt])
#    print(dsvdt)

#    user_in = input('"Enter" to continue or "Ctrl+d" to cancel.')   
#    if user_in == KeyboardInterrupt:
#        sys.exit(0)

    return dsvdt



""" Use integrator to call dsvdt and solve to SS  """
"-----------------------------------------------------------------------------"    
# Create vectors to store outputs:
i_ext = np.hstack([i_ext0, i_ext1, i_ext2])
eta_ss = np.zeros(len(i_ext) +1)
dphi_ss = np.zeros(len(i_ext) +1)
sv_save = np.zeros([len(SV_0) +1, len(i_ext) +1])

# Define common index for last CL node's phi_dl:
iPhi_f = int(iSV['phi_dl'] + (Ny-1)*L_cl/Ny)

# Update and convert i_ext: A/cm^2 -> A/m^2
p['i_ext'] = i_OCV*(100**2) 

sol = solve_ivp(lambda t,sv: dsvdt_func(t, sv, obj, p, ptrs),
      [0, t_sim], SV_0, method=method, atol=atol, rtol=rtol,
      max_step=max_t)

# Store solution and update initial values:
SV_0 = sol.y[:,-1]
sv_save[:,0] = np.append(i_OCV, sol.y[:,-1])
dphi_ss[0] = sol.y[iPhi_f, -1] - dphi_eq_an - p['i_ext']*p['R_naf'] / 100**2
                
print('t_f:', sol.t[-1],'i_ext:', round(i_OCV*1e-4,3),
      'eta:', round(eta_ss[0],3), 'dPhi:', round(dphi_ss[0],3))

if all([i_OCV == 0, polar == 1]):
    for i in range(len(i_ext)):
        
        # Update and convert i_ext: A/cm^2 -> A/m^2
        p['i_ext'] = i_ext[i]*(100**2) 

        sol = solve_ivp(lambda t,sv: dsvdt_func(t, sv, obj, p, ptrs),
              [0, t_sim], SV_0, method=method, atol=atol, rtol=rtol,
              max_step=max_t)
        
        # Store solution and update initial values:
        SV_0 = sol.y[:,-1]
        sv_save[:,i+1] = np.append(i_ext[i], sol.y[:,-1])
    
        eta_ss[i+1] = dphi_ss[0] - sol.y[iPhi_f,-1]
        dphi_ss[i+1] = sol.y[iPhi_f,-1] - dphi_eq_an - p['i_ext']*p['R_naf'] / 100**2

        print('t_f:', sol.t[-1], 'i_ext:', round(p['i_ext']*1e-4,3),
              'eta:', round(eta_ss[i+1],3), 'dPhi:', round(dphi_ss[i+1],3))
        
# If specified, save SS solution at each i_ext:
if save == 1:
    np.savetxt(cwd +'/' +folder_name +'/solution.csv', sv_save, delimiter=',')
