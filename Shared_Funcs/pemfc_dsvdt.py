""" Import needed modules """
"-----------------------------------------------------------------------------"
from scipy.integrate import solve_ivp
from Shared_Funcs.pemfc_transport_funcs import *
import cantera as ct
import numpy as np
import sys

""" Control options for derivative functions """
"-----------------------------------------------------------------------------"
# Toggles to turn on/off inner/outer rxns and gas transports:------------------
pt_rxn = 1
o2_rxn = 1
gas_tog = 1
gdl_tog = 1

""" Define CL dsvdt for core-shell model """
"-----------------------------------------------------------------------------"
def dsvdt_cl_cs(t, sv, dsvdt, objs, p, iSV, gdl_BC):
    """ Set up conditions at GDL/CL BC """
    # Initialize indecies for looping:-----------------------------------------
    cl_ymv = 0  # CL y direction mover (y: GDL -> Elyte)
    
    # Load in BC state and flux from GDL:--------------------------------------
    TDY1 = gdl_BC['TDY1']
    flux_up = gdl_BC['flux_up']
    i_io_up = 0  # no protons flow into the GDL
    
    """ Begin loop - with if statements for CL/Elyte BC """
    for i in range(cl['Ny']):
        # Temperature at each Y node:------------------------------------------
        dsvdt[iSV['T_cl'] +cl_ymv] = 0
        
        # Gas phase species at each Y node:------------------------------------
        if i == cl['Ny'] -1: # BC for CL and electrolyte interface
            flux_dwn = np.zeros(gas_ca.n_species)
        else:
            rho_gas_k = sv[iSV['rho_gas_k'] +cl_ymv +cl['nxt_y']]
            TDY2 = sv[iSV['T_cl'] +cl_ymv +cl['nxt_y']], sum(rho_gas_k), rho_gas_k
            flux_dwn = fickian_adf(TDY1, TDY2, gas_ca, cl, gas_tog)

        # Set the phases for O2 absorption rxn:
        rho_gas_k = sv[iSV['rho_gas_k'] +cl_ymv]
        rho_naf_k = sv[iSV['rho_naf_k'] +cl_ymv]
        
        gas_ca.TDY = sv[iSV['T_cl'] +cl_ymv], sum(rho_gas_k), rho_gas_k
        naf_b_ca.TDY = sv[iSV['T_cl'] +cl_ymv], sum(rho_naf_k), rho_naf_k
        
        rho_dot_g = naf_s_ca.get_net_production_rates(gas_ca) *cl['SApv_naf']\
                  *cl['1/eps_g'] *gas_ca.molecular_weights *gas_tog
        rho_dot_n = naf_s_ca.get_net_production_rates(naf_b_ca) *cl['SApv_naf']\
                  *cl['1/eps_n'] *naf_b_ca.molecular_weights
                  
        # Include rxn and flux in ODE term:        
        dsvdt[iSV['rho_gas_k'] +cl_ymv] = (flux_up - flux_dwn)*cl['1/eps_g']*cl['1/dy']\
                                        + o2_rxn *rho_dot_g

        flux_up = flux_dwn
        TDY1 = TDY2
        
        # Nafion densities at each R node:-------------------------------------
        # The Naftion densities change due to reactions at the outter and inner
        # most shells as well as fluxes between adjacent shells. The direction
        # of storage for the radial terms are done from the outermost shell
        # to the innermost one.
        
        " Start by evaluating the outermost shell "
        # This node contains an O2 absorption rxn with the gas phase as well as
        # a maxx flux with the adjacent inner node.
        rho_k1 = sv[iSV['rho_naf_k'] +cl_ymv]
        rho_k2 = sv[iSV['rho_naf_k'] +cl_ymv +cl['nxt_r']]
        rho_flx_inr = radial_fdiff(rho_k1, rho_k2, cl, 0, ver, 'core_shell')
        
        # Combine absorption and flux to get overall ODE for Nafion densities:
        dsvdt[iSV['rho_naf_k'] +cl_ymv] = o2_rxn *rho_dot_n *cl['1/Vf_shl'][0]\
                                        - rho_flx_inr *cl['1/r_j'][0]**2 *cl['1/t_shl'][0]
                                        
        dsvdt[iSV['rho_naf_k'][cl['iH']] +cl_ymv] = 0      # Ensure constant proton density

        rho_flx_otr = rho_flx_inr
        rho_k1 = rho_k2

        " Evaluate the inner shell nodes "
        for j in range(1, cl['Nr'] -1):
            rho_k2 = sv[iSV['rho_naf_k'] +cl_ymv +(j+1)*cl['nxt_r']]
            rho_flx_inr = radial_fdiff(rho_k1, rho_k2, cl, j, ver, 'core_shell')

            iMid = iSV['rho_naf_k'] +cl_ymv +j*cl['nxt_r']
            dsvdt[iMid] = (rho_flx_otr - rho_flx_inr) *cl['1/r_j'][j]**2 *cl['1/t_shl'][j]
            
            rho_flx_otr = rho_flx_inr
            rho_k1 = rho_k2                                          

        " Apply the Pt reaction BC at the innermost shell "
        # Set the phases for the ORR at the Pt surface:
        carb_ca.electric_potential = 0
        pt_s_ca.electric_potential = 0

        naf_b_ca.electric_potential = -sv[iSV['phi_dl'] +cl_ymv]
        naf_s_ca.electric_potential = -sv[iSV['phi_dl'] +cl_ymv]
        
        pt_s_ca.coverages = sv[iSV['theta_pt_k'] +cl_ymv]
        naf_b_ca.TDY = sv[iSV['T_cl'] +cl_ymv], sum(rho_k1), rho_k1
        rho_dot_n = pt_s_ca.get_net_production_rates(naf_b_ca) *cl['SApv_pt']\
                  *cl['1/eps_n'] *naf_b_ca.molecular_weights
        
        # Pt surface coverages:
        dsvdt[iSV['theta_pt_k'] +cl_ymv] = pt_s_ca.get_net_production_rates(pt_s_ca)\
                                         *cl['1/gamma'] *pt_rxn
        
        # Innermost Nafion node densities:
        iLast = iSV['rho_naf_k'] +cl_ymv +(cl['Nr'] -1)*cl['nxt_r']
        dsvdt[iLast] = pt_rxn *rho_dot_n *cl['1/Vf_shl'][-1] \
                     + rho_flx_otr *cl['1/r_j'][-1]**2 *cl['1/t_shl'][-1]
        
        # Double layer potential at each Y node:-------------------------------
        # The double layer potential is only stored as a function of CL depth.
        # This means that no local potential gradients are shored in the radial
        # direction throughout the Nafion shells.

        # Find ionic currents and define ODE for phi_dl:
        if i == cl['Ny'] -1: # BC for CL and electrolyte interface
            i_io_dwn = cl['i_ext']
        else:
            i_io_dwn = (sv[iSV['phi_dl'] +cl_ymv] - sv[iSV['phi_dl'] +cl_ymv +cl['nxt_y']])\
                     *cl['sig_naf_io'] *cl['1/dy']

        i_Far = pt_rxn *pt_s_ca.get_net_production_rates(carb_ca) *ct.faraday

        i_dl = (i_io_up - i_io_dwn)*cl['1/dy'] - i_Far*cl['SApv_pt']
        dsvdt[iSV['phi_dl'] +cl_ymv] = i_dl*cl['1/CA_dl']

        i_io_up = i_io_dwn

        # Update Y direction moving index:-------------------------------------
        cl_ymv = cl_ymv +cl['nxt_y']
        
    return dsvdt

""" Define CL dsvdt for flooded-agglomerate model """
"-----------------------------------------------------------------------------"
def dsvdt_cl_fa(t, sv, dsvdt, objs, p, iSV, gdl_BC):
    """ Set up conditions at GDL/CL BC """
    # Initialize indecies for looping:-----------------------------------------
    cl_ymv = 0  # CL y direction mover (y: GDL -> Elyte)
    
    # Load in BC state and flux from GDL:--------------------------------------
    TDY1 = gdl_BC['TDY1']
    flux_up = gdl_BC['flux_up']
    i_io_up = 0  # no protons flow into the GDL
    
    """ Begin loop - with if statements for CL/Elyte BC """
    for i in range(cl['Ny']):
        # Temperature at each Y node:------------------------------------------
        dsvdt[iSV['T_cl'] +cl_ymv] = 0

        # Gas phase species at each Y node:------------------------------------        
        if i == cl['Ny'] -1:
            flux_dwn = np.zeros(gas_ca.n_species)
        else:
            rho_gas_k = sv[iSV['rho_gas_k'] +cl_ymv +cl['nxt_y']]
            TDY2 = sv[iSV['T_cl'] +cl_ymv +cl['nxt_y']], sum(rho_gas_k), rho_gas_k
            flux_dwn = fickian_adf(TDY1, TDY2, gas_ca, cl, gas_tog)

        # Set the phases for O2 absorption rxn:
        rho_gas_k = sv[iSV['rho_gas_k'] +cl_ymv]
        rho_shl_k = sv[iSV['rho_shl_k'] +cl_ymv]

        gas_ca.TDY = sv[iSV['T_cl'] +cl_ymv], sum(rho_gas_k), rho_gas_k
        naf_b_ca.TDY = sv[iSV['T_cl'] +cl_ymv], sum(rho_shl_k), rho_shl_k

        rho_dot_g = naf_s_ca.get_net_production_rates(gas_ca) *cl['SApv_naf']\
                  *cl['1/eps_g'] *gas_ca.molecular_weights *gas_tog
        rho_dot_n = naf_s_ca.get_net_production_rates(naf_b_ca) *cl['SApv_naf']\
                  *cl['1/eps_n'] *naf_b_ca.molecular_weights

        # Include rxn and flux in ODE term:
        dsvdt[iSV['rho_gas_k'] +cl_ymv] = (flux_up - flux_dwn)*cl['1/eps_g']*cl['1/dy']\
                                        + o2_rxn *rho_dot_g

        flux_up = flux_dwn
        TDY1 = TDY2

        # Nafion densities at each R node:-------------------------------------
        # The Nafion densities change due to reactions throughout the inner 
        # agglomerate as well as fluxes between adjacent radial nodes. The 
        # direction of storage for the radial terms starts with a single node 
        # for the outer shell, and then continues from the outer agglomerate 
        # node into the center.
                
        " Start by evaluating single-node nafion shell "
        # This node contains an O2 absorption rxn with the gas phase as well as
        # a mass flux with the inner agglomerate.
        rho_k1 = sv[iSV['rho_shl_k'] +cl_ymv]
        rho_k2 = sv[iSV['rho_naf_k'] +cl_ymv]
        rho_flx_inr = radial_fdiff(rho_k1, rho_k2, cl, 0, ver, 'flooded_agg')

        # Combine absorption and flux to get overall ODE:
        dsvdt[iSV['rho_shl_k'] +cl_ymv] = o2_rxn *rho_dot_n - rho_flx_inr

        dsvdt[iSV['rho_shl_k'][cl['iH']] +cl_ymv] = 0      # Ensure constant proton density

        rho_flx_otr = rho_flx_inr
        rho_k1 = rho_k2

        " Evaluate the inner agglomerate nodes "
        # Loop through radial nodes within agglomerate:
        i_Far_r = np.zeros(cl['Nr'])

        # Set the phases for ORR at the Pt surface:
        carb_ca.electric_potential = 0
        pt_s_ca.electric_potential = 0

        naf_b_ca.electric_potential = -sv[iSV['phi_dl'] +cl_ymv]
        naf_s_ca.electric_potential = -sv[iSV['phi_dl'] +cl_ymv]

        for j in range(cl['Nr'] -1):
            rho_k2 = sv[iSV['rho_naf_k'] +cl_ymv +(j+1)*cl['nxt_r']]
            rho_flx_inr = radial_fdiff(rho_k1, rho_k2, cl, j+1, ver, 'flooded_agg')

            pt_s_ca.coverages = sv[iSV['theta_pt_k'] +cl_ymv +j*cl['nxt_r']]
            naf_b_ca.TDY = sv[iSV['T_cl'] +cl_ymv], sum(rho_k1), rho_k1
            rho_dot_n = pt_s_ca.get_net_production_rates(naf_b_ca) *cl['SApv_pt']\
                      *cl['1/eps_n'] *naf_b_ca.molecular_weights *cl['Vf_ishl'][j]

            i_Far_r[j] = pt_rxn *pt_s_ca.get_net_production_rates(carb_ca)\
                       *ct.faraday *cl['Vf_ishl'][j]

            # Pt surface coverages:
            iMid = iSV['theta_pt_k'] +cl_ymv +j*cl['nxt_r']
            dsvdt[iMid] = pt_s_ca.get_net_production_rates(pt_s_ca) *cl['1/gamma'] *pt_rxn
                                        

            # Combine ORR and flux to get overall ODE for Nafion densities:
            iMid = iSV['rho_naf_k'] +cl_ymv +j*cl['nxt_r']
            dsvdt[iMid] = rho_flx_otr - rho_flx_inr + pt_rxn *rho_dot_n

            dsvdt[iMid[cl['iH']]] = 0                     # Ensure constant proton density

            rho_flx_otr = rho_flx_inr
            rho_k1 = rho_k2

        " Apply symmetric flux BC at innermost agglomerate node "
        rho_flx_inr = np.zeros(naf_b_ca.n_species)

        # Set the phases for ORR at the Pt surface:
        pt_s_ca.coverages = sv[iSV['theta_pt_k'] +cl_ymv +(cl['Nr'] -1)*cl['nxt_r']]
        naf_b_ca.TDY = sv[iSV['T_cl'] +cl_ymv], sum(rho_k1), rho_k1
        rho_dot_n = pt_s_ca.get_net_production_rates(naf_b_ca) *cl['SApv_pt']\
                  *cl['1/eps_n'] *naf_b_ca.molecular_weights *cl['Vf_ishl'][-1]

        i_Far_r[-1] = pt_rxn *pt_s_ca.get_net_production_rates(carb_ca)\
                    *ct.faraday *cl['Vf_ishl'][-1]
                    
        # Pt surface coverages:
        iLast = iSV['theta_pt_k'] +cl_ymv +(cl['Nr'] -1)*cl['nxt_r']
        dsvdt[iLast] = pt_s_ca.get_net_production_rates(pt_s_ca) *cl['1/gamma'] *pt_rxn

        # Combine ORR and flux to get overall ODE:
        iLast = iSV['rho_naf_k'] +cl_ymv +(cl['Nr'] -1)*cl['nxt_r']
        dsvdt[iLast] = rho_flx_otr - rho_flx_inr + pt_rxn *rho_dot_n

        dsvdt[iLast[cl['iH']]] = 0                        # Ensure constant proton density

        # Double layer potential at each Y node:-------------------------------
        # The double layer potential is only stored as a function of CL depth,
        # but is based on the reactions that occur throughout the radial
        # direction of each agglomerate. Looping through the radial nodes of
        # each agglomerate and summing over all faradaic currents is used to
        # evaluate an overall double layer current.
        
        " Simplify all radial terms into a single y-dependent double layer "
        # Combine the faradaic currents to get overall i_Far:
        i_Far = np.sum(i_Far_r)

        # Find ionic currents and define ODE for phi_dl:                 
        if i == cl['Ny'] -1:
            i_io_dwn = cl['i_ext']
        else:
            i_io_dwn = (sv[iSV['phi_dl'] +cl_ymv] - sv[iSV['phi_dl'] +cl_ymv +cl['nxt_y']])\
                     *cl['sig_naf_io'] *cl['1/dy']

        i_dl = (i_io_up - i_io_dwn)*cl['1/dy'] - i_Far*cl['SApv_pt']

        dsvdt[iSV['phi_dl'] +cl_ymv] = i_dl*cl['1/CA_dl']

        i_io_up = i_io_dwn

        # Update Y direction moving index:-------------------------------------
        cl_ymv = cl_ymv +cl['nxt_y']
        
    return dsvdt

""" Define dsvdt for pemfc models - common for GDL and then CLs above """
"-----------------------------------------------------------------------------"
def dsvdt_func(t, sv, objs, p, iSV):
    # Initialize indecies for looping:-----------------------------------------
    gdl_ymv = 0 # GDL y direction mover (y: gas channel -> CL)

    dsvdt = np.zeros_like(sv)
    
    """ Bondary Condition - GDL and CL gas transport """
    # Densities/Temp of GDL gas species and CL BC (top):-----------------------
    gas_ca.TPY = gdl['TPY_BC']
    TDY_BC = gas_ca.TDY
    
    # If GDL diffusion is turned on, compare adjacent nodes with ADF flux to 
    # determine the BC composition between the GDL and CL.
    rho_gdl_k = sv[iSV['rho_gdl_k']]
    TDY1 = sv[iSV['T_gdl']], sum(rho_gdl_k), rho_gdl_k
    flux_up = fickian_adf(TDY_BC, TDY1, gas_ca, gdl, gdl_tog)
    
    for k in range(gdl['Ny'] -1):
        rho_gdl_k = sv[iSV['rho_gdl_k'] +gdl_ymv +gdl['nxt_y']]
        TDY2 = sv[iSV['T_gdl'] +gdl_ymv +gdl['nxt_y']], sum(rho_gdl_k), rho_gdl_k
        flux_dwn = fickian_adf(TDY1, TDY2, gas_ca, gdl, gdl_tog)
        
        dsvdt[iSV['rho_gdl_k'] +gdl_ymv] = (flux_up - flux_dwn)*gdl['1/eps_g']*gdl['1/dy']
                                                                                                                                            
        flux_up = flux_dwn
        TDY1 = TDY2
        gdl_ymv = gdl_ymv +gdl['nxt_y']
        
    # Use the composition and state of the last GDL node to calculate the flux
    # into the first CL node.
    rho_gas_k = sv[iSV['rho_gas_k']]
    TDY2 = sv[iSV['T_cl']], sum(rho_gas_k), rho_gas_k
    flux_dwn = fickian_adf(TDY1, TDY2, gas_ca, gdl_cl, gdl_tog)
    
    dsvdt[iSV['rho_gdl_k'] +gdl_ymv] = (flux_up - flux_dwn)*gdl['1/eps_g']*gdl['1/dy']
    
    flux_up = fickian_adf(TDY1, TDY2, gas_ca, gdl_cl, gas_tog)
    TDY1 = TDY2
    
    # Load BC values to pass into CL functions:
    gdl_BC = {}
    gdl_BC['TDY1'] = TDY1
    gdl_BC['flux_up'] = flux_up

    """ Generic loop for interal CL nodes in y-direction """
    if model == 'core_shell':
        dsvdt = dsvdt_cl_cs(t, sv, dsvdt, objs, p, iSV, gdl_BC)
    elif model == 'flooded_agg':
        dsvdt = dsvdt_cl_fa(t, sv, dsvdt, objs, p, iSV, gdl_BC)

#    print(t)
#    print(dsvdt)
#
#    user_in = input('"Enter" to continue or "Ctrl+d" to cancel.')   
#    if user_in == KeyboardInterrupt:
#        sys.exit(0)

    return dsvdt

""" Use integrator to call dsvdt and solve to SS  """
"-----------------------------------------------------------------------------"    
# Create vectors to store outputs:
i_ext = np.hstack([i_OCV, i_ext0, i_ext1, i_ext2])
eta_ss, dphi_ss = np.zeros_like(i_ext), np.zeros_like(i_ext)
sv_save = np.zeros([len(SV_0) +1, len(i_ext)])

# Define common index for last CL node's phi_dl:
iPhi_f = int(iSV['phi_dl'] + (Ny_cl-1)*L_cl/Ny_cl)

# Update and convert i_ext: A/cm^2 -> A/m^2
cl['i_ext'] = i_ext[0] *100**2

sol = solve_ivp(lambda t, sv: dsvdt_func(t, sv, objs, p, iSV), [0, t_sim], 
                SV_0, method=method, atol=atol, rtol=rtol, max_step=max_t)

# Calculate extra PEM resistance terms to subtract off:
R_naf_vec = i_ext*(pem['R_naf'] + 0.5*cl['dy'] / cl['sig_naf_io'] *100**2)

# Store solution and update initial values:
SV_0, sv_save[:,0] = sol.y[:,-1], np.append(i_ext[0], sol.y[:,-1])
dphi_ss[0] = sol.y[iPhi_f, -1] - dphi_eq_an - R_naf_vec[0]
                
print('t_f:',sol.t[-1],'i_ext:',round(cl['i_ext']*1e-4,3), 'dPhi:',round(dphi_ss[0],3))

for i in range(len(i_ext) -1):
    # Don't run the for loop if i_OCV was not set to 0...
    if any([all([i == 0, i_OCV != 0]), polar == 'off']): 
        break
    
    # Update and convert i_ext: A/cm^2 -> A/m^2
    cl['i_ext'] = i_ext[i+1] *100**2

    sol = solve_ivp(lambda t, sv: dsvdt_func(t, sv, objs, p, iSV), [0, t_sim], 
                    SV_0, method=method, atol=atol, rtol=rtol, max_step=max_t)
    
    # Store solution and update initial values:
    SV_0, sv_save[:,i+1] = sol.y[:,-1], np.append(i_ext[i+1], sol.y[:,-1])

    eta_ss[i+1] = dphi_ss[0] - sol.y[iPhi_f,-1]
    dphi_ss[i+1] = sol.y[iPhi_f,-1] - dphi_eq_an - R_naf_vec[i+1]

    print('t_f:',sol.t[-1], 'i_ext:',round(cl['i_ext']*1e-4,3), 'dPhi:',round(dphi_ss[i+1],3))