""" Import needed modules """
"-----------------------------------------------------------------------------"
import numpy as np
import cantera as ct
from Shared_Funcs.pemfc_property_funcs import *  
    
""" Pre-load Phases and Set States """
"-----------------------------------------------------------------------------"
# Cathode Phases:
carb_ca = ct.Solution(ctifile, 'metal')
carb_ca.TP = T_ca, P_ca

gas_ca = ct.Solution(ctifile, 'cathode_gas')
gas_ca.TP = T_ca, P_ca
 
naf_b_ca = ct.Solution(ctifile, 'naf_bulk_ca')
naf_b_ca.TP = T_ca, P_ca

pt_s_ca = ct.Interface(ctifile, 'Pt_surf_ca', [carb_ca, naf_b_ca, gas_ca])
pt_s_ca.TP = T_ca, P_ca

naf_s_ca = ct.Interface(ctifile, 'naf_surf_ca', [naf_b_ca, gas_ca])
naf_s_ca.TP = T_ca, P_ca

# Anode Phases:
carb_an = ct.Solution(ctifile, 'metal')
carb_an.TP = T_an, P_an

gas_an = ct.Solution(ctifile, 'anode_gas')
gas_an.TP = T_an, P_an

naf_b_an = ct.Solution(ctifile, 'naf_bulk_an')
naf_b_an.TP = T_an, P_an

pt_s_an = ct.Interface(ctifile, 'Pt_surf_an', [carb_an, naf_b_an])
pt_s_an.TP = T_an, P_an

naf_s_an = ct.Interface(ctifile, 'naf_surf_an', [naf_b_an, gas_an])
naf_s_an.TP = T_an, P_an

# Change basis to ensure TDY density takes in mass units:
basis = 'mass'

# Set each phase to mass basis:
naf_b_ca.basis = basis
carb_ca.basis = basis
gas_ca.basis = basis
pt_s_ca.basis = basis
naf_s_ca.basis = basis

###############################################################################
# Change parameters for optimization only:
if 'optimize' in vars(): None
else: optimize = 0
    
if optimize == 1:
    if tog == 2:
        R_naf = (c*w_Pt**d + e) *1e-3
    else:
        R_naf = R_naf_opt
        
    theta = theta_opt
        
    Rxn1 = pt_s_ca.reaction(0)
    if tog == 1:
        i_o = a*w_Pt + b
    else:
        i_o = i_o_opt
    Rxn1.rate = ct.Arrhenius(i_o, 0, 0)
    pt_s_ca.modify_reaction(0, Rxn1)
    
    O2 = naf_b_ca.species(naf_b_ca.species_index('O2(Naf)'))
    fuel_coeffs = O2.thermo.coeffs
    fuel_coeffs[[1,8]] = np.array([3.28253784 +offset_opt, 3.782456360 +offset_opt])
    O2.thermo = ct.NasaPoly2(200, 3500, ct.one_atm, fuel_coeffs)
    naf_b_ca.modify_species(naf_b_ca.species_index('O2(Naf)'), O2)
###############################################################################

" Store phases in a common 'objs' dict: "
objs = {}
objs['carb_ca'] = carb_ca
objs['gas_ca'] = gas_ca
objs['naf_b_ca'] = naf_b_ca
objs['pt_s_ca'] = pt_s_ca
objs['naf_s_ca'] = naf_s_ca

objs['carb_an'] = carb_an
objs['gas_an'] = gas_an
objs['naf_b_an'] = naf_b_an
objs['pt_s_an'] = pt_s_an
objs['naf_s_an'] = naf_s_an

""" Pointers for Generality """
"-----------------------------------------------------------------------------"
# Pointers in Solution Vector:
" Gas diffusion layer (GDL) variables "
iSV = {}
iSV['T_gdl'] = 0
iSV['rho_gdl_k'] = np.arange(1, 1 +gas_ca.n_species)

" Catalyst layer (CL) variables "
iSV['T_cl'] = (iSV['rho_gdl_k'][-1] +1)*Ny_gdl
iSV['phi_dl'] = (iSV['rho_gdl_k'][-1] +1)*Ny_gdl +1
iSV['rho_gas_k'] = np.arange(iSV['phi_dl'] +1, iSV['phi_dl'] +1 +gas_ca.n_species)

if model == 'core_shell':
    n_r_species = naf_b_ca.n_species
    iSV['theta_pt_k'] = np.arange(iSV['rho_gas_k'][-1] +1, iSV['rho_gas_k'][-1] +1 +pt_s_ca.n_species)
    iSV['rho_naf_k'] = np.arange(iSV['theta_pt_k'][-1] +1, iSV['theta_pt_k'][-1] +1 +naf_b_ca.n_species)
elif model == 'flooded_agg':
    n_r_species = naf_b_ca.n_species +pt_s_ca.n_species
    iSV['rho_shl_k'] = np.arange(iSV['rho_gas_k'][-1] +1, iSV['rho_gas_k'][-1] +1 +naf_b_ca.n_species)
    iSV['theta_pt_k'] = np.arange(iSV['rho_shl_k'][-1] +1, iSV['rho_shl_k'][-1] +1 +pt_s_ca.n_species)
    iSV['rho_naf_k'] = np.arange(iSV['theta_pt_k'][-1] +1, iSV['theta_pt_k'][-1] +1 +naf_b_ca.n_species)    

""" Pre-Processing and Initialization """
"-----------------------------------------------------------------------------"
" Determine SV length/initialize "
# Initialize all nodes according to cti and user inputs:
gas_ca.TPY = T_ca_BC, P_ca_BC, Y_ca_BC
SV_0_gas_k = gas_ca.density_mass*gas_ca.Y
SV_0_naf_k = naf_b_ca.density_mass*naf_b_ca.Y
SV_0_pt_k = pt_s_ca.coverages

SV_0_gdl = np.tile(np.hstack((T_ca, SV_0_gas_k)), Ny_gdl)

if model == 'core_shell':
    SV_0_rad = np.tile(SV_0_naf_k, Nr_cl) 
    SV_0_cl = np.tile(np.hstack(([T_ca, phi_ca_init], SV_0_gas_k, SV_0_pt_k, SV_0_rad)), Ny_cl)
elif model == 'flooded_agg':
    SV_0_rad = np.tile(np.hstack([SV_0_pt_k, SV_0_naf_k]), Nr_cl)
    SV_0_cl = np.tile(np.hstack(([T_ca, phi_ca_init], SV_0_gas_k, SV_0_naf_k, SV_0_rad)), Ny_cl)

SV_0 = np.zeros(len(SV_0_gdl) + len(SV_0_cl))
SV_0 = np.hstack([SV_0_gdl, SV_0_cl])

L_gdl = int(len(SV_0_gdl))
L_cl = int(len(SV_0_cl))
L_sv = L_gdl + L_cl

" Geometric parameters "
# Calculate all of the surface areas and volume fractions needed. Different
# methods include area_calcs = 0 which assumes that the Pt is flat circles and
# performs calculations with the specified % Pt. Using area_cals = 1 performs
# calculations using Pt-loading information and assumes 1/2 spheres of Pt on 
# the carbon surface. 

# Special cases for area_calcs = 2 and 3 were made for the shell thickness 
# runner of the core-shell model. They assume that as the Nafion shell becomes
# thinner, that the additional volume is given to the gas or carbon phase
# respectively.

""" Core-shell geometric options """
"-----------------------------------------------------------------------------"
if all([model == 'core_shell', area_calcs == 0]):
    # Determine method for Nafion SA based on theta:
    A_naf_reg = 4*np.pi*(r_c+t_naf)**2
    A_naf_edit = 4*np.pi*r_c**2*(p_Pt/100)*(1+np.tan(np.deg2rad(theta)))**2
    
    # Area of naf/gas interface per total volume [m^2 Naf-gas int / m^3 tot]
    if A_naf_reg < A_naf_edit:
        SApv_naf = 3*(1 -eps_g_cl) / (r_c+t_naf)
        p_eff_SAnaf = 1.0
    else:
        SApv_naf = (1 -eps_g_cl) *A_naf_edit / (4/3 *np.pi *(r_c+t_naf)**3)
        p_eff_SAnaf = A_naf_edit / A_naf_reg
        
    # Double layer surface area per total volume [m^2 cathode / m^3 tot]
    SApv_dl = 3*(1 -eps_g_cl) *r_c**2 / (r_c+t_naf)**3
        
    # Pt surface area per total volume [m^2 Pt / m^3 tot]
    SApv_pt = 3*(1 -eps_g_cl) *r_c**2 *(p_Pt/100) / (r_c+t_naf)**3
    
    # Volume fraction of nafion in CL [-]
    eps_n_cl = ((r_c+t_naf)**3 -r_c**3) *(1 -eps_g_cl) / (r_c+t_naf)**3
        
elif all([model == 'core_shell', area_calcs == 1]):
    geom_out = rxn_areas_cs(w_Pt, t_cl, eps_g_cl, t_naf, r_c, r_Pt, rho_Pt, theta)
    
    SApv_naf = geom_out['SApv_naf']
    SApv_dl = geom_out['SApv_dl']
    SApv_pt = geom_out['SApv_pt']    
    eps_n_cl = geom_out['eps_n_cl']
    
    p_Pt = SApv_pt / SApv_dl *100
    p_eff_SAnaf = geom_out['p_eff_SAnaf']
        
elif all([model == 'core_shell', area_calcs == 2]):
    eps_g_cl = 1 - (SApv_dl *(r_c+t_naf)**3 / (3 *r_c**2))
    eps_n_cl = ((r_c+t_naf)**3 -r_c**3) *(1 -eps_g_cl) / (r_c+t_naf)**3
    SApv_naf = 3*(1 -eps_g_cl) / (r_c+t_naf)
        
elif all([model == 'core_shell', area_calcs == 3]):
    r_c = 3*(1 -eps_g_cl) / SApv_naf - t_naf
    p_Pt = SApv_pt *(r_c+t_naf)**3 / (3*(1 -eps_g_cl) *r_c**2) *100
    SApv_dl = SApv_pt / p_Pt
    eps_n_cl = ((r_c+t_naf)**3 -r_c**3) *(1 -eps_g_cl) / (r_c+t_naf)**3

if model == 'core_shell':
    # Print geometry information to console:        
    print('\nt_naf:',t_naf, 'eps_g_cl:',eps_g_cl, 'eps_n_cl:',eps_n_cl)
    print('A_int:',SApv_naf, 'A_dl:',SApv_dl, 'A_Pt:',SApv_pt, '%Pt:',p_Pt)
    
    # Tortuosity calculation via Bruggeman correlation [-]:
    tau_g_gdl = eps_g_gdl**(-0.5)
    tau_g_cl = eps_g_cl**(-0.5)
    tau_n_cl = eps_n_cl**(-0.5)
    
    # Radius vectors for diffusion calculations [m]
    r_j = np.linspace((r_c+t_naf)-t_naf/(2*Nr_cl), r_c+t_naf/(2*Nr_cl), Nr_cl)
    t_shl = np.tile(t_naf/Nr_cl, Nr_cl)
    dr = abs(np.diff(r_j))
    
    r_jph = np.zeros(Nr_cl -1)
    for i in range(Nr_cl -1):
        r_jph[i] = np.mean(r_j[i:i+2])
        
    # Vol fracs of Nafion shells for weighted flux terms [-]:
    Vf_shl = np.zeros(Nr_cl)
    for i in range(Nr_cl):
        Vf_shl[i] = ((r_j[i] +t_shl[i]/2)**3 - (r_j[i] -t_shl[i]/2)**3) \
                  / ((r_c+t_naf)**3 - r_c**3)

""" Flooded-agglomerate geometric options """
"-----------------------------------------------------------------------------"
if all([model == 'flooded-agg', area_calcs == 0]):
    # Area of naf/gas interface per total volume [m^2 Naf-gas int / m^3 tot]
    SApv_naf = 3*(1 -eps_g_cl) / (r_agg+t_naf)

    # Double layer surface area per total volume [m^2 cathode / m^3 tot]
    SApv_dl = 3*(1 -eps_g_cl) *(p_c/100) *r_agg**3 / (r_agg+t_naf)**3 / r_c

    # Pt surface area per total volume [m^2 Pt / m^3 tot]
    SApv_pt = 3*(1 -eps_g_cl) *(p_c/100) *r_agg**3 *(p_Pt/100) / (r_agg+t_naf)**3 / r_c

    # Volume fraction of nafion in CL [-]
    eps_n_cl = (1 -eps_g_cl)*((r_agg+t_naf)**3 - (p_c/100) *r_agg**3) / (r_agg+t_naf)**3

    # Volume fraction of nafion in each inner agglomerate [-]:
    eps_n_agg = 1 - (p_c/100)

    # Volume fraction of nafion in each agglomerate core and shell [-]:
    V_naf_agg = 4/3 *np.pi *(r_agg+t_naf)**3 - 4/3 *np.pi *(r_agg)**3 + 4/3 *np.pi *(r_agg)**3 *eps_n_agg
    Vf_naf_shl = (4/3 *np.pi *(r_agg+t_naf)**3 - 4/3 *np.pi *(r_agg)**3) / V_naf_agg
    Vf_naf_core = 4/3 *np.pi *(r_agg)**3 *eps_n_agg / V_naf_agg
    eps_n_shl = eps_n_cl*Vf_naf_shl
    eps_n_core = eps_n_cl*Vf_naf_core

elif all([model == 'flooded_agg', area_calcs == 1]):
    geom_out = rxn_areas_fa(w_Pt, t_cl, eps_g_cl, r_agg, t_naf, p_c, r_c, r_Pt, rho_Pt)

    SApv_naf = geom_out['SApv_naf']
    SApv_dl = geom_out['SApv_dl']
    SApv_pt = geom_out['SApv_pt']
    eps_n_cl = geom_out['eps_n_cl']
    eps_n_agg = 1 - (p_c/100) #- (geom_out['p_pt_agg']/100)
    V_naf_agg = geom_out['V_naf_agg']

    p_Pt = SApv_pt / SApv_dl

    Vf_naf_shl = (4/3*np.pi*(r_agg +t_naf)**3 - 4/3*np.pi*(r_agg)**3) / V_naf_agg
    Vf_naf_core = 4/3*np.pi*(r_agg)**3*eps_n_agg / V_naf_agg
    eps_n_shl = eps_n_cl*Vf_naf_shl
    eps_n_core = eps_n_cl - eps_n_shl

if model == 'flooded_agg':
    # Ignore any theta dependence when using agglomerate model
    p_eff_SAnaf = 1.0
    
    # Print geometry information to console:
    print('\neps_g:',eps_g_cl, 'eps_n:',eps_n_cl, 'eps_n_agg:',eps_n_agg)
    print('A_int:',SApv_naf, 'A_dl:',SApv_dl, 'A_Pt:',SApv_pt, '%Pt:',SApv_pt/SApv_dl*100)
    
    # Tortuosity calculations via Bruggeman correlation [-]:
    tau_n_agg = 1.05 #eps_n_agg**(-0.5)
    tau_g_gdl = eps_g_gdl**(-0.5)
    tau_g_cl = eps_g_cl**(-0.5)
    tau_n_cl = eps_n_cl**(-0.5)
    
    # Radius vectors for diffusion calculations [m]:
    r_j = np.hstack([r_agg +t_naf/2, np.linspace(r_agg -r_agg/(2*Nr_cl), r_agg/(2*Nr_cl), Nr_cl)])
    t_shl = np.hstack([t_naf, np.tile(r_agg/Nr_cl, Nr_cl)])
    dr = abs(np.diff(r_j))
    
    r_jph = np.zeros(Nr_cl)
    r_jph[0] = r_agg
    for i in range(Nr_cl -1):
        r_jph[i+1] = np.mean(r_j[i+1:i+2])
        
    # Vol fracs of Nafion shells for weighted drho_dt terms [-]:
    Vf_shl = np.zeros(Nr_cl +1)
    Vf_shl[0] = eps_n_shl / eps_n_cl
    for i in range(Nr_cl):
        Vf_shl[i+1] = ((r_j[i+1] +t_shl[i+1]/2)**3 - (r_j[i+1] -t_shl[i+1]/2)**3)\
                    / r_agg**3 *(1 - Vf_shl[0])
    
    # Vol fracs of inner agglomerate for weighted faradaic terms [-]:
    Vf_ishl = np.zeros(Nr_cl)
    for i in range(Nr_cl):
        Vf_ishl[i] = ((r_j[i+1] +t_shl[i+1]/2)**3 - (r_j[i+1] -t_shl[i+1]/2)**3) / r_agg**3

" Calculate the anode equilibrium potential for polarization curves "
dgibbs_an = pt_s_an.delta_gibbs
dphi_eq_an = -dgibbs_an / (n_elec_an*ct.faraday)

" Initialize dictionaries to pass parameters to functions "
gdl, cl, gdl_cl, pem, p = {}, {}, {}, {}, {}

" Load the GDL parameters into a dictionary "
gdl['wt1'] = 0.5
gdl['wt2'] = 0.5
gdl['dy'] = t_gdl / Ny_gdl
gdl['1/dy'] = 1 / gdl['dy']
gdl['1/eps_g'] = 1 / eps_g_gdl
gdl['K_g'] = 6e-12 / 0.75 *eps_g_gdl  # scale permeability by values from [3]
gdl['eps/tau2'] = eps_g_gdl / tau_g_gdl**2
gdl['y'] = t_gdl
gdl['Ny'] = Ny_gdl
gdl['nxt_y'] = int(L_gdl / Ny_gdl)    # spacing between adjacent y nodes in GDL
gdl['TPY_BC'] = T_ca_BC, P_ca_BC, Y_ca_BC

" Load the CL parameters into a dictionary "
cl['wt1'] = 0.5
cl['wt2'] = 0.5
cl['dy'] = t_cl / Ny_cl
cl['1/dy'] = 1 / cl['dy']
cl['1/eps_g'] = 1 / eps_g_cl
cl['1/eps_n'] = 1 / eps_n_cl
cl['K_g'] = 8e-16 / 0.4 *eps_g_cl     # scale permeability by values from [3]
cl['eps/tau2'] = eps_g_cl / tau_g_cl**2
cl['eps/tau2_n'] = eps_n_cl #/ tau_n_cl**2 # based on full cell Nafion vol frac
cl['eps/tau2_n2'] = 1 # ignore for cs model, overwrite below for fa
cl['y'] = t_cl
cl['Ny'] = Ny_cl
cl['Nr'] = Nr_cl
cl['nxt_y'] = int(L_cl / Ny_cl)       # spacing between adjacent y nodes in CL
cl['nxt_r'] = n_r_species             # spacing between adjacent r nodes in CL
cl['i_e'] = ind_e
cl['iH'] = naf_b_ca.species_index('H(Naf)')
cl['SApv_pt'] = SApv_pt
cl['SApv_naf'] = SApv_naf
cl['1/CA_dl'] = 1 / (C_dl*SApv_dl) 
cl['p_eff_SAnaf'] = p_eff_SAnaf
cl['D_eff_naf'] = D_eff_naf_func(T_ca, t_naf, p_Pt, cl, D_O2_method, model)
cl['sig_naf_io'] = sig_naf_io_func(T_ca, t_naf, RH, p_Pt, cl, sig_method, model)
cl['1/r_j'] = 1 / r_j 
cl['r_jph'] = r_jph
cl['1/t_shl'] = 1 / t_shl
cl['1/dr'] = 1 / dr
cl['Vf_shl'] = Vf_shl
cl['1/Vf_shl'] = 1 / Vf_shl
cl['1/gamma'] = 1 / pt_s_ca.site_density
    
if model == 'flooded_agg':
    cl['eps/tau2_n2'] = eps_n_agg / tau_n_agg**2 # inner agglomerate only
    cl['Vf_ishl'] = Vf_ishl
    cl['1/Vf_ishl'] = 1 / Vf_ishl

" Calculate BC parameters between GDL and CL "
gdl_cl['dy'] = 0.5*gdl['dy'] + 0.5*cl['dy']
gdl_cl['1/dy'] = 1 / gdl_cl['dy']
gdl_cl['wt1'] = 0.5*gdl['dy'] / gdl_cl['dy']
gdl_cl['wt2'] = 0.5*cl['dy'] / gdl_cl['dy']
gdl_cl['K_g'] = gdl_cl['wt1']*gdl['K_g'] + gdl_cl['wt2']*cl['K_g']
gdl_cl['eps/tau2'] = gdl_cl['wt1']*gdl['eps/tau2'] + gdl_cl['wt2']*cl['eps/tau2']

" Load any PEM parameters into a dictionary "
pem['R_naf'] = R_naf

" Combine all dictionaries "
p['model'] = model
p['gdl'] = gdl
p['gdl_cl'] = gdl_cl
p['cl'] = cl
p['pem'] = pem     