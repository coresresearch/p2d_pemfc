""" Import needed modules """
"-----------------------------------------------------------------------------"
import os, sys, csv
import numpy as np
import cantera as ct
from shutil import copy2, rmtree
from pemfc_property_funcs import *
#from read_write_funcs import *



""" Set up saving location """
"-----------------------------------------------------------------------------"
cwd = os.getcwd()

# Create folder for any files/outputs to be saved:
if all([os.path.exists(folder_name), save == 1]):
    print('\nWARNING: folder_name already exists. Files will be overwritten.')
    print('\n"Enter" to continue and overwrite or "Ctrl+c" to cancel.')
    print('In a GUI, e.g. Spyder, "Ctrl+d" may be needed to cancel.')
    user_in = input()   
    if user_in == KeyboardInterrupt:
        sys.exit(0)
    else:
        rmtree(folder_name)
        os.makedirs(folder_name)
        copy2(cwd + '/particle_shell_pemfc_runner.py', folder_name)
        copy2(cwd + '/particle_shell_pemfc_pre.py', folder_name)
        copy2(cwd + '/particle_shell_pemfc_dsvdt.py', folder_name)
        copy2(cwd + '/particle_shell_pemfc_post.py', folder_name)
        copy2(cwd + '/pemfc_transport_funcs.py', folder_name)
        copy2(cwd + '/pemfc_property_funcs.py', folder_name)
        ModuleWriter(cwd + '/' + folder_name + '/user_inputs.csv', 
                     __import__('particle_shell_pemfc_runner'))
        
elif save == 1:
    os.makedirs(folder_name)
    copy2(cwd + '/particle_shell_pemfc_runner.py', folder_name)
    copy2(cwd + '/particle_shell_pemfc_pre.py', folder_name)
    copy2(cwd + '/particle_shell_pemfc_dsvdt.py', folder_name)
    copy2(cwd + '/particle_shell_pemfc_post.py', folder_name)
    copy2(cwd + '/pemfc_transport_funcs.py', folder_name)
    copy2(cwd + '/pemfc_property_funcs.py', folder_name)
    ModuleWriter(cwd + '/' + folder_name + '/user_inputs.csv', 
                 __import__('particle_shell_pemfc_runner'))

# Save the current cti files into new folder:
cti_path = ct.__path__[0]
if all([os.path.exists(ctifile), save == 1]):
    copy2(ctifile, folder_name)
elif save == 1:
    copy2(cti_path + '/data/' + ctifile, folder_name)
    
    
    
""" Pre-load Phases and Set States """
"-----------------------------------------------------------------------------"
# Cathode Phases:
carb_ca = ct.Solution(ctifile, 'metal')
carb_ca.TP = T_ca, P_ca

gas_ca = ct.Solution(ctifile, 'cathode_gas')
gas_ca.TP = T_ca, P_ca

naf_b_ca = ct.Solution(ctifile, 'naf_bulk_ca')
naf_b_ca.TP = T_ca, P_ca

pt_s_ca = ct.Interface(ctifile, 'Pt_surf_ca', [carb_ca, naf_b_ca])
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

" For ease and clarity, let's store phases in a common 'obj' dict: "
obj = {} # Done for passing phases to external functions (ex: dSVdt)
obj['carb_ca'] = carb_ca
obj['gas_ca'] = gas_ca
obj['naf_b_ca'] = naf_b_ca
obj['pt_s_ca'] = pt_s_ca
obj['naf_s_ca'] = naf_s_ca

obj['carb_an'] = carb_an
obj['gas_an'] = gas_an
obj['naf_b_an'] = naf_b_an
obj['pt_s_an'] = pt_s_an
obj['naf_s_an'] = naf_s_an



""" Pointers for Generality """
"-----------------------------------------------------------------------------"
# Pointers in Solution Vector:
iSV = {}
iSV['phi_dl'] = 0
iSV['T'] = 1
v_1d = len(iSV)
# put all 1D CL state variables above ^

iSV['rho_n1'] = np.arange(v_1d, v_1d + naf_b_ca.n_species-1)
iSV['rho_naf_k'] = np.arange(v_1d, v_1d + (naf_b_ca.n_species-1)*Nr)

iSV['rho_gas_k'] = np.arange(iSV['rho_naf_k'][-1] +1,
                             iSV['rho_naf_k'][-1] +1 +gas_ca.n_species)

iSV['T_gdl'] = (iSV['rho_gas_k'][-1] +1)*Ny
iSV['rho_gdl_k'] = np.arange(iSV['T_gdl'] +1, iSV['T_gdl'] +1 +gas_ca.n_species)

# Pointers in Production Rates (H+ neglected since constant density):
iPt = {}
iPt['carb'] = np.arange(0, carb_ca.n_species)
iPt['Naf'] = np.arange(iPt['carb'][-1] +2, iPt['carb'][-1] +1 +naf_b_ca.n_species)
iPt['int'] = np.arange(iPt['Naf'][-1] +1, len(pt_s_ca.net_production_rates))

iNaf = {}
iNaf['Naf'] = np.arange(0 +1, naf_b_ca.n_species)
iNaf['gas'] = np.arange(iNaf['Naf'][-1] +1, iNaf['Naf'][-1] +1 +gas_ca.n_species)
iNaf['int'] = np.arange(iNaf['gas'][-1] +1, len(naf_s_ca.net_production_rates))

# Dictionary to pass all pointers to dSVdt function:
ptrs = {}
ptrs['iSV'] = iSV
ptrs['iNaf'] = iNaf
ptrs['iPt'] = iPt



""" Pre-Processing and Initialization """
"-----------------------------------------------------------------------------"
" Determine SV length/initialize "
SV_0 = np.zeros((v_1d + (naf_b_ca.n_species -1)*Nr + gas_ca.n_species)*Ny
                + (1 + gas_ca.n_species)*Ny_gdl)

# Change basis to ensure TDY density takes in mass units:
basis = 'mass'

# Set each phase to mass basis:
naf_b_ca.basis = basis
carb_ca.basis = basis
gas_ca.basis = basis
pt_s_ca.basis = basis
naf_s_ca.basis = basis

# Set point for cathode gas phase BC (O2 flow channel):
TPY_ca_BC = T_ca_BC, P_ca_BC, Y_ca_BC

# Initialize all nodes according to cti and user inputs:
SV_0_v_1d = [phi_ca_init, gas_ca.T]
SV_0_naf_k = np.tile(naf_b_ca.density_mass*naf_b_ca.Y[iNaf['Naf']], Nr)

gas_ca.TPY = TPY_ca_BC
SV_0_gas_k = gas_ca.density_mass*gas_ca.Y
SV_0_cl = np.tile(np.hstack((SV_0_v_1d, SV_0_naf_k, SV_0_gas_k)), Ny)
SV_0_gdl = np.tile(np.hstack((gas_ca.T, SV_0_gas_k)), Ny_gdl)

SV_0 = np.hstack([SV_0_cl, SV_0_gdl])

L_cl = int(len(SV_0_cl))
L_gdl = int(len(SV_0_gdl))

" Geometric parameters "
if area_calcs == 0: # Calcs for areas based on % Pt and flat circles
    # Determine method for Nafion SA based on theta:
    A_naf_reg = 4*np.pi*(r_c+t_naf)**2
    A_naf_edit = 4*np.pi*r_c**2*(p_Pt/100)*(1+np.tan(np.deg2rad(theta)))**2
    
    # Area of naf/gas interface per total volume [m^2 Naf-gas int / m^3 tot]
    if A_naf_reg < A_naf_edit:
        SA_pv_naf = 3*(1 - eps_gas) / (r_c+t_naf)
        p_eff_SAnaf = 1.0
    else:
        SA_pv_naf = (1 - eps_gas)*A_naf_edit / (4/3*np.pi*(r_c+t_naf)**3)
        p_eff_SAnaf = A_naf_edit / A_naf_reg
        
    # Pt surface area per total volume [m^2 Pt / m^3 tot]
    SA_pv_pt = 3*(1 - eps_gas)*r_c**2*(p_Pt/100) / (r_c+t_naf)**3
    
    # Cathode surface area per total volume [m^2 cathode / m^3 tot]
    SA_pv_dl = 3*(1 - eps_gas)*r_c**2 / (r_c+t_naf)**3
    
    # Volume fraction of nafion [-]
    eps_naf = ((r_c+t_naf)**3 - r_c**3)*(1 - eps_gas) / (r_c+t_naf)**3
    
    print('\nt_naf:',t_naf, 'Vf_gas:',eps_gas, 'Vf_naf:',eps_naf, '%Pt:',p_Pt)
    print('A_int:',SA_pv_naf, 'A_dl:',SA_pv_dl, 'A_Pt:',SA_pv_pt)
    
elif area_calcs == 1: # Calcs for areas based on Pt-loading and 1/2 spheres
    geom_out = rxn_areas(w_Pt, t_cl, eps_gas, t_naf, r_c, r_Pt, rho_Pt, theta)
    
    SA_pv_naf = geom_out['SA_pv_naf']
    SA_pv_pt = geom_out['SA_pv_pt']
    SA_pv_dl = geom_out['SA_pv_dl']
    eps_naf = geom_out['eps_naf']
    
    p_Pt = SA_pv_pt / SA_pv_dl *100
    p_eff_SAnaf = geom_out['p_eff_SAnaf']
    
    print('\nt_naf:',t_naf, 'Vf_gas:',eps_gas, 'Vf_naf:',eps_naf, '%Pt:',p_Pt)
    print('A_int:',SA_pv_naf, 'A_dl:',SA_pv_dl, 'A_Pt:',SA_pv_pt)
    
elif area_calcs == 2: # Assume only gas phase changes as t_naf changes
    eps_gas = 1 - (SA_pv_dl *(r_c + t_naf)**3 / (3 *r_c**2))
    eps_naf = ((r_c+t_naf)**3 - r_c**3)*(1-eps_gas) / (r_c+t_naf)**3
    SA_pv_naf = 3*(1 - eps_gas) / (r_c+t_naf)
    
    print('\nt_naf:',t_naf, 'Vf_gas:',eps_gas, 'Vf_naf:',eps_naf, '%Pt:',p_Pt)
    print('A_int:',SA_pv_naf, 'A_dl:',SA_pv_dl, 'A_Pt:',SA_pv_pt)
    
elif area_calcs == 3: # Assume only carbon phase changes as t_naf changes
    r_c = 3*(1 - eps_gas) / SA_pv_naf - t_naf
    p_Pt = SA_pv_pt*(r_c+t_naf)**3 / (3*(1 - eps_gas)*r_c**2) *100
    SA_pv_dl = SA_pv_pt / p_Pt
    eps_naf = ((r_c+t_naf)**3 - r_c**3)*(1 - eps_gas) / (r_c+t_naf)**3
        
    print('\nt_naf:',t_naf, 'Vf_gas:',eps_gas, 'Vf_naf:',eps_naf, '%Pt:',p_Pt)
    print('A_int:',SA_pv_naf, 'A_dl:',SA_pv_dl, 'A_Pt:',SA_pv_pt)

# Tortuosity calculation via Bruggeman correlation [-]:
tau_gas = eps_gas**(-0.5)
tau_gdl = eps_gdl**(-0.5)
tau_naf = eps_naf**(-0.5)

# Radius vectors for diffusion calculations [m]
r_j = np.linspace(r_c+t_naf/(2*Nr), (r_c+t_naf)-t_naf/(2*Nr), Nr)
t_shl = np.tile(t_naf/Nr, Nr)
dr = np.diff(r_j)

r_jph = np.zeros(Nr-1)
for i in range(Nr-1):
    r_jph[i] = np.mean(r_j[i:i+2])
    
# Volume fractions of inner agglomerate shells for weighted faradaic rxns [-]:
Vf_shl = np.zeros(Nr)
for i in range(Nr):
    Vf_shl[i] = ((r_j[i] +t_shl[i]/2)**3 - (r_j[i] -t_shl[i]/2)**3) \
              / ((r_c+t_naf)**3 - r_c**3)

" Calculate the anode equilibrium potential for polarization curves "
dgibbs_an = pt_s_an.delta_gibbs
dphi_eq_an = -dgibbs_an / (n_elec_an*ct.faraday)

" Let's load the peters into a peters dictionary "
gdl = {}
gdl['wt1'] = 0.5
gdl['wt2'] = 0.5
gdl['dy'] = t_gdl / Ny_gdl
gdl['1/dy'] = 1 / gdl['dy']
gdl['K_g'] = 6e-12 / 0.75 *eps_gdl  # scale permeability by values from [3]
gdl['eps/tau2'] = eps_gdl / tau_gdl**2

p = {} 
p['gdl'] = gdl
p['wt1'] = 0.5
p['wt2'] = 0.5
p['Ny_gdl'] = Ny_gdl
p['Ny'] = Ny
p['Nr'] = Nr
p['t_cl'] = t_cl
p['ind_e'] = ind_e
p['TPY_gas_ca_BC'] = TPY_ca_BC
p['SV_0'] = SV_0
p['SA_pv_pt'] = SA_pv_pt
p['SA_pv_naf'] = SA_pv_naf
p['K_g'] = 8e-16 / 0.4 *eps_gas     # scale permeability by values from [3]
p['eps/tau2'] = eps_gas / tau_gas**2
p['eps/tau2_n'] = eps_naf #/ tau_naf**2
p['p_eff_SAnaf'] = p_eff_SAnaf
p['D_eff_naf'] = D_eff_naf_func(naf_b_ca.T, t_naf, p_Pt, p, D_O2_method)
p['sig_naf_io'] = sig_naf_io_func(naf_b_ca.T, t_naf, RH, p_Pt, p, sig_method)
p['rho_H'] = naf_b_ca.density_mass*naf_b_ca.Y[0]
p['dy'] = t_cl / Ny
p['r_jph'] = r_jph
p['R_naf'] = R_naf

" For simplicity, calculate boundary condition terms between GDL and CL "
gdl_cl = {}
gdl_cl['dy'] = 0.5*gdl['dy'] + 0.5*p['dy']
gdl_cl['1/dy'] = 1 / gdl_cl['dy']
gdl_cl['wt1'] = 0.5*gdl['dy'] / gdl_cl['dy']
gdl_cl['wt2'] = 0.5*p['dy'] / gdl_cl['dy']
gdl_cl['K_g'] = gdl_cl['wt1']*gdl['K_g'] + gdl_cl['wt2']*p['K_g']
gdl_cl['eps/tau2'] = gdl_cl['wt1']*gdl['eps/tau2'] + gdl_cl['wt2']*p['eps/tau2']
                     
" For speed, calculate inverses/division terms "
p['gdl_cl'] = gdl_cl
p['1/Ny'] = 1 / Ny
p['1/dr'] = 1 / dr
p['1/r_j'] = 1 / r_j 
p['1/Vf_shl'] = 1 / Vf_shl
p['1/dy'] = Ny / t_cl
p['1/t_cl'] = 1 / t_cl
p['1/t_shl'] = 1 / t_shl
p['1/eps_naf'] = 1 / eps_naf
p['1/eps_gas'] = 1 / eps_gas
p['1/eps_gdl'] = 1 / eps_gdl
p['1/(C*A)'] = 1 / (C_dl*SA_pv_dl)                

" Save paters in dictionary to file for post processing "
if save == 1:    
    from read_write_funcs import *
    DictWriter(cwd + '/' + folder_name + '/params.csv', p)

" Save current solution if running as post only "    
if all([save == 1, post_only == 1]):
    np.savetxt(cwd +'/' +folder_name +'/solution.csv', sv_save, delimiter=',')
