""" Model Description """
"-----------------------------------------------------------------------------"
"""This model is a particle/shell model for a PEM fuel cell that allows for
pseudo-2D transport analysis through the thin film of nafion surrounding
each carbon particle. The model is only written for the cathode side, but
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

""" Import needed modules """
"-----------------------------------------------------------------------------"
import numpy as np
import cantera as ct

""" User Input Parameters """
"-----------------------------------------------------------------------------"
ctifile = 'pemfc.cti'

" Initial electrochemical values "
i_OCV = 0                         # 0 for OCV [A/cm^2] -> o.w. single run
i_ext0 = np.linspace(0,0.1,10)    # external currents close to 0 [A/cm^2]
i_ext1 = np.linspace(0.1,1,10)    # external currents after 0 [A/cm^2]
i_ext2 = np.linspace(1,1.5,10)    # external currents further from 0 [A/cm^2]
phi_ca_init = 0.7                 # initial cathode potential [V]
T_ca, P_ca = 350, ct.one_atm      # cathode temp [K] and pressure [Pa] at t = 0
T_an, P_an = 350, ct.one_atm      # anode temp [K] and pressure [Pa] at t = 0

" Transport and material properties "
C_dl = 1.5e-6        # capacitance of double layer [F/m^2]
sig_naf_io = 7.5     # ionic conductivity of Nafion [S/m]
D_eff_naf = 8.45e-10 # effective diffusion coefficients in nafion phase [m^2/s]

" Pt loading and geometric values "
r_c = 50e-9/2       # radius of single carbon particle [m]
t_naf = 5e-9        # thickness of nafion around a carbon particle [m]
t_ca = 15e-6        # thickness of cathode modeled in simulation [m]
p_Pt = 10           # percentage of Pt covering the carbon particle surface [%]
eps_gas = 0.5       # volume fraction of gas [-]

" Gas channel boundary conditions "
Y_ca_BC = 'N2: 0.79, O2: 0.21, Ar: 0.01, H2O: 0.03, OH: 0'
T_ca_BC, P_ca_BC = 350, ct.one_atm  # cathode gas channel T [K] and P [Pa]

" Reaction properties "
n_elec_an = 2       # sum(nu_k*z_k) for anode surface reaction from ctifile

" Model Inputs "
index_electron = 0  # index of electrons in Pt surface... from cti
method = 'BDF'      # method for solve_ivp [eg: BDF,RK45,LSODA,Radau, etc...]
t_sim = 1e2         # time span of integration [s]
Ny = 5              # number of spacial discretizations for catalyst layer
Nr = 3              # number of radial discretizations for nafion film

" Modify Tolerance for convergence "
max_step = t_sim    # maximum allowable time step for solver [s]
atol = 1e-9         # absolute tolerance passed to solver
rtol = 1e-6         # relative tolerance passed to solver

" Plot toggles - ('0' off and '1' on) "
post_only = 0       # turn on to only run post-processing plots
debug = 1           # turn on to plot double layer cathode / rho_naf vs t
radial = 0          # turn on radial O2 plots for each Nafion shell
grads = 0           # turn on to plot O2 and Phi gradients in depth of cathode
polar = 1           # turn on to generate full cell polarization curves
over_p = 0          # turn on to plot overpotential curve for cathode side

" Saving options "
folder_name = 'folder_name'     # folder name for saving all files/outputs
save = 1                        # toggle saving on/off with '1' or '0'
