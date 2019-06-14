""" Model Description """
"-----------------------------------------------------------------------------"
"""This model is a a flooded-agglomerate model for a PEM fuel cell that allows
for pseudo-2D transport analysis through the agglomerates of C/Pt particles
covered in nafion. The model is only written for the cathode side, but
could easily be expanded to incorporate the anode."""

""" Model Assumptions """
"-----------------------------------------------------------------------------"
" 1.) Gas properties are constant -> parameter"
" 2.) Surface sites are constant -> parameter"
" 3.) Temperature is constant -> parameter"
" 4.) Variables are Phi_dl [V] -> only difference important, temp [K],"
"     Nafion/gas species mass densities, rho_naf_k and rho_gas_k [kg/m^3]"
" 5.) The proton density in Nafion is constant due to its structure"
" 6.) The C/Pt phase at every node has the same potential"
" 7.) The ionic conductivity of Nafion is a function of its RH, temperature,"
"     and thickness and can be modeled via various sub models described in"
"     detail in the pemfc_property_func file."

""" Import needed modules """
"-----------------------------------------------------------------------------"
import os
import cantera as ct
import numpy as np
import pylab as plt
from scipy.integrate import solve_ivp
from Shared_Funcs.pemfc_transport_funcs import *
from Shared_Funcs.pemfc_property_funcs import *
ct.add_directory(os.getcwd() + '/Flooded_Agg')

""" User Input Parameters """
"-----------------------------------------------------------------------------"
ctifile = 'pemfc_fa.cti'

" Initial electrochemical values "
i_OCV = 0                           # 0 [A/cm^2] -> or single if polar == 0
i_ext0 = np.linspace(0.001,0.1,5)   # external currents close to 0 [A/cm^2]
i_ext1 = np.linspace(0.101,1.0,5)   # external currents [A/cm^2]
i_ext2 = np.linspace(1.100,2.0,5)   # external currents further from 0 [A/cm^2]
phi_ca_init = 0.7                   # initial cathode potential [V]
T_ca, P_ca = 333, 1.5*ct.one_atm    # cathode temp [K] and pressure [Pa] at t = 0
T_an, P_an = 333, 1.5*ct.one_atm    # anode temp [K] and pressure [Pa] at t = 0

" Transport and material properties "
RH = 95              # relative humidity of Nafion phase [%]
C_dl = 1.5e-9        # capacitance of double layer [F/m^2]
sig_method = 'sun'   # 'lam', 'bulk', 'mix', or 'sun' for conductivity method
D_O2_method = 'sun'  # 'lam', 'bulk', 'mix', or 'sun' for D_O2 method
R_naf = 60e-3        # resistance of Nafion membrane [Ohm*cm^2]

" Pt loading and geometric values "
area_calcs = 1      # control for area calculations [0:p_Pt, 1:Pt_loading]
p_Pt = 20           # percentage of Pt covering the carbon particle surface [%]
p_c = 95            # % of C sheres packed in agglomerate - Gauss max = 74%
w_Pt = 0.2          # loading of Pt on carbon [mg/cm^2]
rho_Pt = 21.45e3    # density of Pt for use in area property calcs [kg/m^3]
r_c = 25e-9         # radius of single carbon particle [m]
r_Pt = 1e-9         # radius of Pt 1/2 sphere sitting on C surface [m]
r_agg = 50e-9       # radius of agglomerate excluding nafion shell [m]
t_naf = 5e-9        # thickness of nafion shell around each agglomerate [m]
t_gdl = 250e-6      # thickness of cathode GDL modeled in simulation [m]
t_cl = 15e-6        # thickness of cathode CL modeled in simulation [m]
eps_g_gdl = 0.5     # porosity of GDL [-]
eps_g_cl = 0.1      # porosity of CL [-]

" Gas channel boundary conditions "
Y_ca_BC = 'N2: 0.79, O2: 0.21, Ar: 0.01, H2O: 0.03, OH: 0'
T_ca_BC, P_ca_BC = 333, 1.5*ct.one_atm  # cathode gas channel T [K] and P [Pa]

" Reaction properties "
n_elec_an = 2       # sum(nu_k*z_k) for anode surface reaction from ctifile

" Model Inputs "
ind_e = 0           # index of electrons in Pt surface phase... from cti
method = 'BDF'      # method for solve_ivp [eg: BDF,RK45,LSODA,Radau,etc...]
t_sim = 1e2         # time span of integration [s]
Ny_gdl = 3          # number of depth discretizations for GDL
Ny_cl = 5           # number of depth discretizations for CL
Nr_cl = 5           # number of radial discretizations for CL inner agglomerate

" Modify Tolerance for convergence "
max_t = t_sim       # maximum allowable time step for solver [s]
atol = 1e-9         # absolute tolerance passed to solver
rtol = 1e-6         # relative tolerance passed to solver

" Plot toggles - (0: off and 1: on) "
post_only = 0       # turn on to only run post-processing
debug = 1           # turn on to plot first node variables vs time
radial = 1          # turn on radial O2 plots for each agglomerate
grads = 1           # turn on to plot O2 and Phi gradients in depth of cathode
polar = 1           # turn on to generate full cell polarization curves
over_p = 1          # turn on to plot overpotential curve for cathode side

i_val = 1           # validate current between GDL and CL with O2 flux calcs
i_cl = 0            # plot the fraction of i_Far against CL depth at i_find
i_find = 0.5        # current from polarization curve to use in i_val verification
ver = 2             # temporary debugging tool for radial diffusion

" Plotting options "
font_nm = 'Arial'   # name of font for plots
font_sz = 14        # size of font for plots

" Saving options "
folder_name = 'folder_name'     # folder name for saving all files/outputs
save = 0                        # toggle saving on/off with '1' or '0'

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

# Change basis to ensure TDY density takes in mass units:
basis = 'mass'

# Set each phase to mass basis:
naf_b_ca.basis = basis
carb_ca.basis = basis
gas_ca.basis = basis
pt_s_ca.basis = basis
naf_s_ca.basis = basis

" Store phases in a common 'objs' dict: "
obj = {}
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
" Gas diffusion layer (GDL) variables "
iSV = {}
iSV['T_gdl'] = 0
iSV['rho_gdl_k'] = np.arange(1, 1 +gas_ca.n_species)

" Catalyst layer (CL) variables "
iSV['T_cl'] = (iSV['rho_gdl_k'][-1] +1)*Ny_gdl
iSV['phi_dl'] = (iSV['rho_gdl_k'][-1] +1)*Ny_gdl +1
iSV['rho_gas_k'] = np.arange(iSV['phi_dl'] +1, iSV['phi_dl'] +1 +gas_ca.n_species)
iSV['rho_shl_k'] = np.arange(iSV['rho_gas_k'][-1] +1, iSV['rho_gas_k'][-1] +1 +naf_b_ca.n_species)
iSV['rho_naf_k'] = np.arange(iSV['rho_shl_k'][-1] +1, iSV['rho_shl_k'][-1] +1 +naf_b_ca.n_species)

""" Pre-Processing """
"-----------------------------------------------------------------------------"
" Determine SV length/initialize "
# Initialize all nodes according to cti and user inputs:
gas_ca.TPY = T_ca_BC, P_ca_BC, Y_ca_BC
SV_0_gas_k = gas_ca.density_mass*gas_ca.Y
SV_0_naf_k = np.tile(naf_b_ca.density_mass*naf_b_ca.Y, Nr_cl +1)

SV_0_gdl = np.tile(np.hstack((T_ca, SV_0_gas_k)), Ny_gdl)
SV_0_cl = np.tile(np.hstack(([T_ca, phi_ca_init], SV_0_gas_k, SV_0_naf_k)), Ny_cl)

SV_0 = np.zeros(len(SV_0_gdl) + len(SV_0_cl))
SV_0 = np.hstack([SV_0_gdl, SV_0_cl])

L_gdl = int(len(SV_0_gdl))
L_cl = int(len(SV_0_cl))
L_sv = L_gdl + L_cl

" Geometric parameters "
if area_calcs == 0: # Calcs for areas based on % Pt and flat circles
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

elif area_calcs == 1: # Calcs for areas based on Pt-loading and 1/2 spheres
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
    r_jph[i+1] = np.mean(r_j[i+1:i+3])
    
# Volume fractions of Nafion shells for weighted drho_dt terms [-]:
Vf_shl = np.zeros(Nr_cl +1)
Vf_shl[0] = eps_n_shl / eps_n_cl
for i in range(Nr_cl):
    Vf_shl[i+1] = ((r_j[i+1] +t_shl[i+1]/2)**3 - (r_j[i+1] -t_shl[i+1]/2)**3)\
                / r_agg**3 *(1 - Vf_shl[0])

# Volume fractions of inner agglomerate for weighted faradaic terms [-]:
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

" Load the CL parametrs into a dictionary "
cl['wt1'] = 0.5
cl['wt2'] = 0.5
cl['dy'] = t_cl / Ny_cl
cl['1/dy'] = 1 / cl['dy']
cl['1/eps_g'] = 1 / eps_g_cl
cl['1/eps_n'] = 1 / eps_n_cl
cl['K_g'] = 8e-16 / 0.4 *eps_g_cl     # scale permeability by values from [3]
cl['eps/tau2'] = eps_g_cl / tau_g_cl**2
cl['eps/tau2_n'] = eps_n_cl #/ tau_n_cl**2 # based on full cell Nafion vol frac
cl['eps/tau2_n2'] = eps_n_agg / tau_n_agg**2 # inner agglomerate only
cl['y'] = t_cl
cl['Ny'] = Ny_cl
cl['Nr'] = Nr_cl
cl['nxt_y'] = int(L_cl / Ny_cl)       # spacing between adjacent y nodes in CL
cl['nxt_r'] = int(naf_b_ca.n_species) # spacing between adjacent r nodes in CL
cl['i_e'] = ind_e
cl['iH'] = naf_b_ca.species_index('H(Naf)')
cl['SApv_pt'] = SApv_pt
cl['SApv_naf'] = SApv_naf
cl['1/CA_dl'] = 1 / (C_dl*SApv_dl)
cl['p_eff_SAnaf'] = 1
cl['D_eff_naf'] = D_eff_naf_func(T_ca, t_naf, p_Pt, cl, D_O2_method, 'flooded_agg')
cl['sig_naf_io'] = sig_naf_io_func(T_ca, t_naf, RH, p_Pt, cl, sig_method, 'flooded_agg')
cl['1/r_j'] = 1 / r_j
cl['r_jph'] = r_jph
cl['1/t_shl'] = 1 / t_shl
cl['1/dr'] = 1 / dr
cl['Vf_shl'] = Vf_shl
cl['1/Vf_shl'] = 1 / Vf_shl
cl['Vf_ishl'] = Vf_ishl

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
p['gdl'] = gdl
p['gdl_cl'] = gdl_cl
p['cl'] = cl
p['pem'] = pem

""" Define ODE for Solution """
"-----------------------------------------------------------------------------"
def dsvdt_func(t, sv, objs, p, iSV):
    # Toggles to turn on/off inner/outer rxns and gas transports:--------------
    pt_rxn = 1
    o2_rxn = 1
    gas_tog = 1
    gdl_tog = 1

    # Extract dictionaries for readability:------------------------------------
    carb_ca = objs['carb_ca']
    gas_ca = objs['gas_ca']
    naf_b_ca = objs['naf_b_ca']
    pt_s_ca = objs['pt_s_ca']
    naf_s_ca = objs['naf_s_ca']

    gdl = p['gdl']
    gdl_cl = p['gdl_cl']
    cl = p['cl']

    # Initialize indecies for looping:-----------------------------------------
    gdl_ymv = 0 # GDL y direction mover (y: gas channel -> CL)
    cl_ymv = 0  # CL y direction mover (y: GDL -> Elyte)

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

    """ Before loop, set BC for CL and GDL Boundary """
    # Ionic current at BC (top) - no protons flow into the GDL:----------------
    i_io_up = 0

    """ Generic loop for interal CL nodes in y-direction """
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

            naf_b_ca.TDY = sv[iSV['T_cl'] +cl_ymv], sum(rho_k1), rho_k1
            rho_dot_n = pt_s_ca.get_net_production_rates(naf_b_ca) *cl['SApv_pt']\
                      *cl['1/eps_n'] *naf_b_ca.molecular_weights *cl['Vf_ishl'][j]

            i_Far_r[j] = pt_rxn *pt_s_ca.get_net_production_rates(carb_ca)\
                       *ct.faraday *cl['Vf_ishl'][j]

            # Combine ORR and flux to get overall ODE:
            iMid = iSV['rho_naf_k'] +cl_ymv +j*cl['nxt_r']
            dsvdt[iMid] = rho_flx_otr - rho_flx_inr + pt_rxn *rho_dot_n

            dsvdt[iMid[cl['iH']]] = 0                     # Ensure constant proton density

            rho_flx_otr = rho_flx_inr
            rho_k1 = rho_k2

        " Apply symmetric flux BC at innermost agglomerate node "
        rho_flx_inr = np.zeros(naf_b_ca.n_species)

        # Set the phases for ORR at the Pt surface:
        naf_b_ca.TDY = sv[iSV['T_cl'] +cl_ymv], sum(rho_k1), rho_k1
        rho_dot_n = pt_s_ca.get_net_production_rates(naf_b_ca) *cl['SApv_pt']\
                  *cl['1/eps_n'] *naf_b_ca.molecular_weights *cl['Vf_ishl'][-1]

        i_Far_r[-1] = pt_rxn *pt_s_ca.get_net_production_rates(carb_ca)\
                    *ct.faraday *cl['Vf_ishl'][-1]

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

#        print(t)
#        print(dsvdt)
#    
#        user_in = input('"Enter" to continue or "Ctrl+d" to cancel.')
#        if user_in == KeyboardInterrupt:
#            sys.exit(0)

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

sol = solve_ivp(lambda t,sv: dsvdt_func(t, sv, obj, p, iSV), [0, t_sim], 
                SV_0, method=method, atol=atol, rtol=rtol, max_step=max_t)

# Calculate extra PEM resistance terms to subtract off:
R_naf = i_ext*(pem['R_naf'] + 0.5*cl['dy'] / cl['sig_naf_io'] *100**2)

# Store solution and update initial values:
SV_0, sv_save[:,0] = sol.y[:,-1], np.append(i_ext[0], sol.y[:,-1])
dphi_ss[0] = sol.y[iPhi_f, -1] - dphi_eq_an - R_naf[0]
                
print('t_f:',sol.t[-1],'i_ext:',round(cl['i_ext']*1e-4,3), 'dPhi:',round(dphi_ss[0],3))

for i in range(len(i_ext) -1):
    # Don't run the for loop if i_OCV was not set to 0...
    if all([i == 0, i_OCV != 0]): 
        break
    
    # Update and convert i_ext: A/cm^2 -> A/m^2
    cl['i_ext'] = i_ext[i+1] *100**2

    sol = solve_ivp(lambda t,sv: dsvdt_func(t, sv, obj, p, iSV), [0, t_sim], 
                    SV_0, method=method, atol=atol, rtol=rtol, max_step=max_t)
    
    # Store solution and update initial values:
    SV_0, sv_save[:,i+1] = sol.y[:,-1], np.append(i_ext[i+1], sol.y[:,-1])

    eta_ss[i+1] = dphi_ss[0] - sol.y[iPhi_f,-1]
    dphi_ss[i+1] = sol.y[iPhi_f,-1] - dphi_eq_an - R_naf[i+1]

    print('t_f:',sol.t[-1], 'i_ext:',round(cl['i_ext']*1e-4,3), 'dPhi:',round(dphi_ss[i+1],3))

""" Post-processing for Plotting and Other Results """
"-----------------------------------------------------------------------------"
# Update plot settings:
font = plt.matplotlib.font_manager.FontProperties(family=font_nm, size=font_sz)
plt.rcParams.update({'font.size': font_sz})

# Generalize figure numbering:
fig_num = 1

# Species indexes:
iO2_g = gas_ca.species_index('O2')
iO2_n = naf_b_ca.species_index('O2(Naf)')
iH_n = naf_b_ca.species_index('H(Naf)')

if debug == 1:
    # Extract cathode double layer potential and plot:
    phi_dl = sol.y[iSV['phi_dl'], :]

    plt.figure(fig_num)
    plt.plot(sol.t, phi_dl)

    plt.ylabel(r'Cathode Double Layer Voltage, $\phi_{dl}$ [V]')
    plt.xlabel('Time, t [s]')
    plt.tight_layout()

    fig_num = fig_num +1

    # Extract nafion species densities and plot:
    legend_str = []
    legend_count = 0

    for i in range(Ny_cl):
        ind1 = iSV['rho_shl_k'][iO2_n]
        ind2 = iSV['rho_shl_k'][iO2_n] +i*cl['nxt_y'] +(Nr_cl +1)*cl['nxt_r']
        species_k_Naf = sol.y[ind1:ind2:naf_b_ca.n_species, :]

        for j in range(Nr_cl +1):
            plt.figure(fig_num)
            plt.plot(sol.t, species_k_Naf[j])

            legend_str.append(naf_b_ca.species_names[iO2_n] +'r' +str(legend_count))
            legend_count = legend_count +1

        plt.title('y-node = ' + str(i))
        plt.legend(legend_str)
        plt.ylabel(r'Nafion Phase Mass Densities, $\rho$ [kg/m$^3$]')
        plt.xlabel('Time, t [s]')
        plt.tight_layout()

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
    plt.ylabel(r'CL Gas Phase Mass Densities, $\rho$ [kg/m$^3$]')
    plt.xlabel('Time, t [s]')
    plt.tight_layout()

    plt.figure(fig_num+1)
    plt.legend(gas_ca.species_names)
    plt.ylabel(r'GDL Gas Phase Mass Densities, $\rho$ [kg/m$^3$]')
    plt.xlabel('Time, t [s]')
    plt.tight_layout()

    fig_num = fig_num +2

if radial == 1:
    # Extract O2 density from each agglomerate as f(r):
    legend_str = []
    legend_count = 0

    for i in range(Ny_cl):
        ind_1 = int(iSV['rho_shl_k'][iO2_n] +i*cl['nxt_y'])
        ind_2 = int(iSV['rho_shl_k'][iO2_n] +i*cl['nxt_y'] +(Nr_cl +1)*cl['nxt_r'])
        naf_O2_r = sol.y[ind_1:ind_2:cl['nxt_r'], -1]

        ind_1 = int(iSV['rho_shl_k'][iH_n] +i*cl['nxt_y'])
        ind_2 = int(iSV['rho_shl_k'][iH_n] +i*cl['nxt_y'] +(Nr_cl +1)*cl['nxt_r'])
        naf_H_r = sol.y[ind_1:ind_2:cl['nxt_r'], -1]

        plt.figure(fig_num)
        plt.plot(1/cl['1/r_j']*1e6, naf_O2_r*1e4, '-o')
        #plt.plot(1/cl['1/r_j']*1e6, naf_H_r, '-o')
        legend_str.append('y node = ' + str(legend_count))
        legend_count = legend_count +1

    plt.xlabel('Nafion Shell Radius, [um]')
    plt.ylabel('Nafion Phase - O2 Density *1e4 [$kg/m^3$]')
    plt.legend(legend_str, loc='best')
    plt.tight_layout()

    fig_num = fig_num +1

if grads == 1:
    # Make sure potential gradient is in correct direction:
    plt.figure(fig_num)
    Phi_elyte_y = -1*sol.y[iSV['phi_dl']:L_sv:cl['nxt_y'], -1]
    plt.plot(np.linspace(0,t_cl*1e6,Ny_cl), Phi_elyte_y, '-o')

    plt.xlabel(r'Cathode CL Depth [$\mu$m]')
    plt.ylabel('Electrolyte Potential [V]')
    plt.tight_layout()

    fig_num = fig_num +1

    # Make sure O2 gradient is in correct direction - gas phase:
    plt.figure(fig_num)

    rho_O2_gdl = sol.y[iSV['rho_gdl_k'][iO2_g]:L_gdl:gdl['nxt_y'], -1]
    rho_O2_CL = sol.y[iSV['rho_gas_k'][iO2_g]:L_sv:cl['nxt_y'], -1]
    rho_O2_y = np.hstack([rho_O2_gdl, rho_O2_CL])

    dist_ca = np.hstack([np.linspace(0.5*gdl['dy'], t_gdl-0.5*gdl['dy'],Ny_gdl),
                         np.linspace(t_gdl+0.5*cl['dy'], t_gdl+t_cl-0.5*cl['dy'],Ny_cl)])

    plt.plot(dist_ca*1e6, rho_O2_y, '-o')

    plt.xlabel(r'Cathode Depth [$\mu$m]')
    plt.ylabel('Gas Phase O2 Density [$kg/m^3$]')
    plt.tight_layout()

    fig_num = fig_num +1

    # Check O2 gradient direction - naf phase (agglomerate shell):
    plt.figure(fig_num)
    rho_O2_y = sol.y[int(iSV['rho_shl_k'][iO2_n]):L_sv:cl['nxt_y'], -1]
    plt.plot(np.linspace(0,t_cl*1e6,Ny_cl), rho_O2_y*1e4, '-o')

    plt.xlabel(r'Cathode CL Depth [$\mu$m]')
    plt.ylabel('Agglomerate shell - O2 Density *1e4 [$kg/m^3$]')
    plt.tight_layout()

    fig_num = fig_num +1

if over_p == 1:
    # Plot a overpotential curve (i_ext vs eta_ss) for the cathode:
    plt.figure(fig_num)
    plt.plot(i_ext, eta_ss)

    plt.ylabel(r'Steady-state Overpotential, $\eta_{ss}$ [V]')
    plt.xlabel(r'External Current, $i_{ext}$ [$A/cm^2$]')
    plt.tight_layout()

    fig_num = fig_num +1

if polar == 1:
    # Plot a polarization curve (i_ext vs dphi_ss) for the cell:
    plt.figure(fig_num)
    plt.plot(i_ext, dphi_ss, linewidth=2, color='C0', linestyle='-')

    plt.ylabel(r'Cell Potential [V]')
    plt.xlabel(r'Current Density [A/cm$^2$]')
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
        plt.errorbar(x, y, yerr=yerr, fmt='o', color=color, capsize=3)
        plt.ylim([0.35, 1.0])
        plt.xlim([0, 3.5])
    except:
        None

    fig_num = fig_num +1

#if i_cl == 1:
#    i_all = np.hstack([i_OCV, i_ext])
#    i_ind = np.argmin(abs(i_all - i_find))
#    sv = sv_save[1:, i_ind]
#
#    # Set up storage location and CL depths:
#    i_Far_cl = np.zeros(p['Ny'])
#    y_cl = np.linspace(t_gdl+0.5*cl['dy'], t_gdl+t_cl-0.5*cl['dy'], Ny_cl)
#
#    # Start with the first node (y: GDL -> Elyte, r: in -> out)
#    sv_mv = 0
#    sv_nxt = int(L_cl / cl['Ny'])
#
#    for j in range(cl['Ny']):
#        # Set the inner shell state:---------------------------------------
#        phi_dl = sv[iSV['phi_dl']+sv_mv]
#        carb_ca.electric_potential = 0
#        pt_s_ca.electric_potential = 0
#
#        naf_b_ca.electric_potential = -phi_dl
#        naf_s_ca.electric_potential = -phi_dl
#
#        rho_naf_k = np.hstack((p['rho_H'], sv[iSV['rho_n1']+sv_mv]))
#
#        T_ca = sv[iSV['T']+sv_mv]
#        naf_b_ca.TDY = T_ca, sum(rho_naf_k), rho_naf_k
#
#        # i_Far [A/cm^2] at each CL depth:---------------------------------
#        sdot_e = pt_s_ca.net_production_rates[cl['i_e']]
#        i_Far_cl[j] = sdot_e*ct.faraday*cl['SApv_pt']
#
#        sv_mv = sv_mv + sv_nxt
#
#    plt.figure(fig_num)
#    plt.plot(y_cl*1e6, -i_Far_cl / (i_ext*cl['1/dy']))
#
#    plt.xlabel(r'Cathode Depth [$\mu$m]')
#    plt.ylabel('$i_{Far}$ / $i_{ext}$ [-]')
#    plt.legend(leg1)
#    plt.tight_layout()
#
#    plt.gca().set_prop_cycle(None)
#
#    fig_num = fig_num +1

plt.show()

if i_val == 1:
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
