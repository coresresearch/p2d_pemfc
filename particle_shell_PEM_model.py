#   This model is a particle/shell model for a PEM fuel cell that allows for 
# pseudo-2D transport analysis through the thin film of nafion surrounding
# each carbon particle. The model is only written for the cathode side, but 
# could easily be expanded to incorporate the anode.

import cantera as ct
import numpy as np
import pylab as plt
from scipy.integrate import solve_ivp

""" Model Assumptions """
"-----------------------------------------------------------------------------"
" 1.) Gas properties are constant -> parameter"
" 2.) Surface sites are constant -> parameter"
" 3.) Temperature is constant -> parameter"
" 4.) State variables are Phi_dl [V] -> only difference important, temp [K],"
"     Nafion/gas species mass densities, rho_naf_k and rho_gas_k [kg/m^3]"
" 5.) The proton density in Nafion is constant due to its structure"
" 6.) The C/Pt phase at every node has the same potential"

""" User Input Parameters """
"-----------------------------------------------------------------------------"
ctifile = 'pemfc.cti'

" Initial electrochemical values "
i_ext1 = np.linspace(0,0.1,20)  # external currents close to 0 [A/cm^2]
i_ext2 = np.linspace(0.1,3,20)  # external currents further from 0 [A/cm^2]
phi_ca_init = 0.7               # initial cathode potential [V]
T_ca, P_ca = 300, ct.one_atm    # cathode temp [K] and pressure [Pa] at t = 0
T_an, P_an = 300, ct.one_atm    # anode temp [K] and pressure [Pa] at t = 0

" Transport and material properties "
C_dl = 1.5e-6       # capacitance of double layer [F/m^2]
sig_naf_io = 7.8    # ionic conductivity of Nafion [S/m]
D_eff_naf = 4.2e-8  # effective diffusion coefficients in nafion phase [m^2/s]

" Pt loading and geometric values "
r_c = 50e-9/2       # radius of single carbon particle [m]
t_naf = 10e-9       # thickness of nafion around a carbon particle [m]
t_ca = 50e-6        # thickness of cathode modeled in simulation [m]
p_Pt = 10           # percentage of Pt covering the carbon particle surface [%]
eps_gas = 0.5       # volume fraction of gas [-]

" Gas channel boundary conditions "
Y_ca_BC = 'N2: 0.79, O2: 0.21, Ar: 0.01, H2O: 0.03, OH: 0'
T_ca_BC, P_ca_BC = 300, ct.one_atm  # cathode gas channel T [K] and P [Pa]

" Reaction properties "
n_elec_an = 2       # sum(nu_k*z_k) for anode surface reaction

" Model Inputs"
index_electron = 0  # index of electrons in Pt surface... from cti
method = 'BDF'      # method for solve_ivp [eg: BDF,RK45,LSODA,Radau, etc...]
t_sim = 1e2         # time span of integration [s]
Ny = 5              # number of spacial discretizations for catalyst layer
Nr = 5              # number of radial discretizations for nafion film

" Modify Tolerance for convergence "
atol = 1e-7         # absolute tolerance passed to solver
rtol = 1e-5         # relative tolerance passed to solver

" Plot toggles - ('0' off and '1' on) "
post_only = 0       # turn on to only run post-processing plots
debug = 0           # turn on to plot double layer cathode / rho_naf vs t
radial = 0          # turn on radial O2 plots for each Nafion shell
grads = 0           # turn on to plot O2 and Phi gradients in depth of cathode
polar = 1           # turn on to generate full cell polarization curves
over_p = 0          # turn on to plot overpotential curve for cathode side

""" Pre-load Phases and Set States """
"-----------------------------------------------------------------------------"
# Cathode Phases:
metal_ca = ct.Solution(ctifile,'metal')
metal_ca.TP = T_ca, P_ca

gas_ca = ct.Solution(ctifile,'cathode_gas')
gas_ca.TP = T_ca, P_ca
Y_ca_BC = gas_ca.Y

naf_bulk_ca = ct.Solution(ctifile,'naf_bulk_ca')
naf_bulk_ca.TP = T_ca, P_ca

Pt_surf_ca = ct.Interface(ctifile,'Pt_surf_ca',[metal_ca,naf_bulk_ca])
Pt_surf_ca.TP = T_ca, P_ca

naf_surf_ca = ct.Interface(ctifile,'naf_surf_ca',[naf_bulk_ca,gas_ca])
naf_surf_ca.TP = T_ca, P_ca

# Anode Phases:
metal_an = ct.Solution(ctifile,'metal')
metal_an.TP = T_an, P_an

gas_an = ct.Solution(ctifile,'anode_gas')
gas_an.TP = T_an, P_an

naf_bulk_an = ct.Solution(ctifile,'naf_bulk_an')
naf_bulk_an.TP = T_an, P_an

Pt_surf_an = ct.Interface(ctifile,'Pt_surf_an',[metal_an,naf_bulk_an])
Pt_surf_an.TP = T_an, P_an

naf_surf_an = ct.Interface(ctifile,'naf_surf_an',[naf_bulk_an,gas_an])
naf_surf_an.TP = T_an, P_an

" For ease and clarity, let's store these in a common 'obj' dict: "
obj = {} # Done for passing phases to external functions (ex: dSVdt)
obj['metal_ca'] = metal_ca
obj['gas_ca'] = gas_ca
obj['naf_bulk_ca'] = naf_bulk_ca
obj['Pt_surf_ca'] = Pt_surf_ca
obj['naf_surf_ca'] = naf_surf_ca

obj['metal_an'] = metal_an
obj['gas_an'] = gas_an
obj['naf_bulk_an'] = naf_bulk_an
obj['Pt_surf_an'] = Pt_surf_an
obj['naf_surf_an'] = naf_surf_an

""" Pointers for Generality """
"-----------------------------------------------------------------------------"
# Pointers in Solution Vector:
SVptr = {}
SVptr['phi_dl'] = 0
SVptr['temp'] = 1
state_vars = len(SVptr) # put all single-node state variables above ^

SVptr['rho_n1'] = np.arange(state_vars, state_vars+naf_bulk_ca.n_species-1)

SVptr['rho_naf_k'] = np.arange(state_vars,
                     state_vars+(naf_bulk_ca.n_species-1)*Nr)

SVptr['rho_gas_k'] = np.arange(state_vars+(naf_bulk_ca.n_species-1)*Nr,
                     state_vars+(naf_bulk_ca.n_species-1)*Nr+gas_ca.n_species)

# Pointers in Production Rates:
Ptptr = {}
Ptptr['metal'] = np.arange(0, metal_ca.n_species)
Ptptr['Naf'] = np.arange(metal_ca.n_species+1,
                          metal_ca.n_species+naf_bulk_ca.n_species) # +1 neg H+
Ptptr['int'] = np.arange(metal_ca.n_species+naf_bulk_ca.n_species,
               len(Pt_surf_ca.net_production_rates))

Nafptr = {}
Nafptr['Naf'] = np.arange(0+1, naf_bulk_ca.n_species) # +1 neglects H+
Nafptr['gas'] = np.arange(naf_bulk_ca.n_species,
                           naf_bulk_ca.n_species+gas_ca.n_species)
Nafptr['int'] = np.arange(naf_bulk_ca.n_species+gas_ca.n_species,
                len(naf_surf_ca.net_production_rates))

# Dictionary to pass all pointers to dSVdt function:
ptrs = {}
ptrs['SVptr'] = SVptr
ptrs['Nafptr'] = Nafptr
ptrs['Ptptr'] = Ptptr

""" Pre-Processing and Initialization """
"-----------------------------------------------------------------------------"
" Determine SV length/initialize "
SV_0 = np.zeros((state_vars+(naf_bulk_ca.n_species-1)*Nr+gas_ca.n_species)*Ny)

# Change basis to ensure density reports mass:
basis = 'mass'
naf_bulk_ca.basis = basis

# Set other phases to basis:
metal_ca.basis = basis
gas_ca.basis = basis
Pt_surf_ca.basis = basis
naf_surf_ca.basis = basis

# Initialize all nodes according to cti and user inputs:
SV_0_state_vars = [phi_ca_init, gas_ca.T]
SV_0_naf_rhos = np.tile(naf_bulk_ca.density*naf_bulk_ca.Y[Nafptr['Naf']], Nr)
SV_0_gas_rhos = gas_ca.density*gas_ca.Y
SV_0 = np.tile(np.hstack((SV_0_state_vars, SV_0_naf_rhos, SV_0_gas_rhos)), Ny)

# Set point for cathode gas phase BC (O2 flow channel):
TPY_ca_BC = T_ca_BC, P_ca_BC, Y_ca_BC

" Geometric parameters "
# Area of naf/gas interface per total volume [m^2 Naf-gas int / m^3 tot]
A_pv_naf_gas_int = 3*(1 - eps_gas) / (r_c + t_naf)

# Pt surface area per total volume [m^2 Pt / m^3 tot]
A_pv_surf_Pt = 3*(1 - eps_gas)*r_c**2*p_Pt / (r_c + t_naf)**3

# Cathode surface area per total volume [m^2 cathode / m^3 tot]
A_pv_ca_naf_int = 3*(1 - eps_gas)*r_c**2 / (r_c + t_naf)**3

# Volume fraction of nafion [-]
eps_naf = ((r_c+t_naf)**3 - r_c**3)*(1-eps_gas) / (r_c+t_naf)**3

# Tortuosity calculation via Bruggeman correlation:
tau_gas = eps_gas**(-0.5) 

# Radius vectors for diffusion calculations [m]
r_nodes = np.linspace(r_c+t_naf/(Nr+1), (r_c+t_naf)-t_naf/(Nr+1), Nr)
t_shells = np.tile(t_naf/Nr, Nr)
delta_rs = np.diff(r_nodes)

r_bounds = np.zeros(Nr-1)
for i in range(Nr-1):
    r_bounds[i] = np.mean(r_nodes[i:i+2])
    
" Calculate the anode equilibrium potential for polarization curves "
Delta_gibbs_an = Pt_surf_an.delta_gibbs
Delta_Phi_eq_an = -Delta_gibbs_an / (n_elec_an*ct.faraday)

" Let's load the parameters into a parameters dictionary "
param = {} # This allows values to be cleanly passed into external functions
param['SV_0'] = SV_0
param['D_eff_naf'] = D_eff_naf
param['sig_naf_io'] = sig_naf_io
param['index_electron'] = index_electron
param['A_pv_surf_Pt'] = A_pv_surf_Pt
param['A_pv_naf_gas_int'] = A_pv_naf_gas_int
param['TPY_gas_ca_BC'] = TPY_ca_BC
param['rho_gas_elyte_BC'] = np.zeros(gas_ca.n_species)
param['H_density'] = naf_bulk_ca.density*naf_bulk_ca.Y[0]
param['r_bounds'] = r_bounds
param['Ny'] = Ny
param['Nr'] = Nr

" For speed, calculate inverses/division terms "
param['eps_naf_inv'] = 1 / eps_naf
param['eps_gas_inv'] = 1 / eps_gas
param['Ny_inv'] = 1 / Ny
param['t_ca_inv'] = 1 / t_ca
param['1/dy'] = Ny / t_ca
param['(C*A)_inv'] = 1 / (C_dl*A_pv_ca_naf_int)
param['phi/tau_sq'] = eps_gas / tau_gas**2
param['1/delta_rs'] = 1 / delta_rs
param['1/t_shells'] = 1 / t_shells
param['1/r_nodes'] = 1 / r_nodes

""" Run Solver to Obtain Solution """
"-----------------------------------------------------------------------------"
if post_only != 1:
    
    from dSVdt_particle_shell_test import dSVdt_func
    
    i_ext = np.hstack([i_ext1,i_ext2])
    eta_SS = np.zeros_like(i_ext)
    Delta_Phi_SS = np.zeros_like(i_ext)
    
    param['i_ext'] = 0*(100**2) # conversion: A/cm^2 -> A/m^2
        
    sol = solve_ivp(lambda t,SV: dSVdt_func(t, SV, obj, param, ptrs),
          [0, t_sim], SV_0, method=method, atol=atol, rtol=rtol)
    
    SV_0 = sol.y[:,-1]
    Phi_0 = sol.y[int(SVptr['phi_dl']+(Ny-1)*len(SV_0)/Ny), -1]
    
    for i in range(len(i_ext)):
        param['i_ext'] = i_ext[i]*(100**2) # conversion: A/cm^2 -> A/m^2
        
        sol = solve_ivp(lambda t,SV: dSVdt_func(t, SV, obj, param, ptrs),
              [0, t_sim], SV_0, method=method, atol=atol, rtol=rtol)
        
        SV_0 = sol.y[:,-1]
        eta_SS[i] = Phi_0 - sol.y[int(SVptr['phi_dl']+(Ny-1)*len(SV_0)/Ny), -1]
        Delta_Phi_SS[i] = sol.y[int(SVptr['phi_dl']+(Ny-1)*len(SV_0)/Ny), -1]\
                        - Delta_Phi_eq_an
        
        print('eta:', round(eta_SS[i],5), 
              'Delta_Phi:', round(Delta_Phi_SS[i],5))

""" Post-processing for Plotting and Other Results """
"-----------------------------------------------------------------------------"
fig_num = 1

if debug == 1:
    # Extract Cathode Double Layer Voltage and Plot:
    phi_dl = sol.y[SVptr['phi_dl'],:]
    
    plt.figure(fig_num)
    plt.plot(sol.t,phi_dl)
    
    plt.ylabel(r'Cathode Voltage, $\phi_{dl}$ [V]')
    plt.xlabel('Time, t [s]')
    plt.show()
    
    fig_num = fig_num +1
    
    # Extract Nafion Species Densities and Plot:
    legend_str = []
    legend_count = 1
    
    for i in range(Ny):
        species_k_Naf = sol.y[SVptr['rho_naf_k']+i*int(len(SV_0)/Ny),:]
        
        for j in range(Nr):
            plt.figure(fig_num)
            plt.plot(sol.t,species_k_Naf[j])
        
            legend_str.append(naf_bulk_ca.species_names[1] + str(legend_count))
            legend_count = legend_count +1
            
        plt.title('y-node =' + str(i))
        plt.legend(legend_str)
        plt.ylabel(r'Mass Density, $\rho$ [kg/m$^3$]')
        plt.xlabel('Time, t [s]')
        plt.show()
        
        fig_num = fig_num +1
        
if radial == 1:
    # Extract O2 density from each Nafion shell as f(r):
    legend_str = []
    legend_count = 1
    
    for i in range(Ny):
        ind_1 = int(SVptr['rho_n1']+i*len(SV_0)/Ny)
        ind_2 = int(SVptr['rho_n1']+i*len(SV_0)/Ny + Nr)
        naf_k_r = sol.y[ind_1:ind_2, -1]
        
        plt.figure(fig_num)
        plt.plot(r_nodes*1e9, naf_k_r*1e4)
        legend_str.append('y node = ' + str(legend_count))
        legend_count = legend_count +1
        
    plt.xlabel('Nafion Shell Radius, [nm]')
    plt.ylabel('Nafion Phase - O2 Density *1e4 [$kg/m^3$]')
    plt.legend(legend_str, loc='lower right')
        
    fig_num = fig_num +1

if grads == 1:
    # Make sure potential gradient is in correct direction:
    plt.figure(fig_num)
    Phi_elyte_y = sol.y[SVptr['phi_dl']:len(SV_0):int(len(SV_0)/Ny),-1]
    plt.plot(np.linspace(0,t_ca*1e6,Ny), Phi_elyte_y, '-o')
    
    plt.xlabel(r'Cathode Depth [$\mu$m]')
    plt.ylabel('Cathode Potential [V]')
    
    fig_num = fig_num +1
        
    # Make sure O2 gradient is in correct direction - gas phase:
    plt.figure(fig_num)
    O2_ind = gas_ca.species_index('O2')
    rho_O2_y = sol.y[SVptr['rho_gas_k'][O2_ind]:len(SV_0):int(len(SV_0)/Ny),-1]
    plt.plot(np.linspace(0,t_ca*1e6,Ny), rho_O2_y, '-o')
    
    plt.xlabel(r'Cathode Depth [$\mu$m]')
    plt.ylabel('Gas Phase - O2 Density [$kg/m^3$]')
    
    fig_num = fig_num +1
    
    # Make sure O2 gradient is in correct direction - nafion phase:
    plt.figure(fig_num)
    rho_O2_y = sol.y[int(SVptr['rho_n1']):len(SV_0):int(len(SV_0)/Ny),-1]
    plt.plot(np.linspace(0,t_ca*1e6,Ny), rho_O2_y*1e4, '-o')
    
    plt.xlabel(r'Cathode Depth [$\mu$m]')
    plt.ylabel('Nafion Phase - O2 Density *1e4 [$kg/m^3$]')
    
    fig_num = fig_num +1
        
if over_p == 1:
    # Plot a overpotential curve (i_ext vs eta_SS) for the cathode:
    plt.figure(fig_num)
    plt.plot(i_ext, eta_SS)
    
    plt.ylabel(r'Steady-state Overpotential, $\eta_{ss}$ [V]')
    plt.xlabel(r'External current, $i_{ext}$ [$A/cm^2$]')
    
    fig_num = fig_num +1
    
if polar == 1:
    # Plot a polarization curve (i_ext vs Delta_Phi_SS) for the cell:
    plt.figure(fig_num)
    plt.plot(i_ext, Delta_Phi_SS)
    
    plt.ylabel(r'Steady-state Potential, $\Delta \Phi$ [V]')
    plt.xlabel(r'External current, $i_{ext}$ [$A/cm^2$]')
    
    fig_num = fig_num +1
