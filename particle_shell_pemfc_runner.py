""" Model Description """
"-----------------------------------------------------------------------------"
"""This model is a particle-shell model for a PEM fuel cell that allows for
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
" 7.) The ionic conductivity of Nafion is a function of its RH, temperature,"
"     and thickness and can be interpolated from experiments reported in a"
"     paper by Paul, McCreery, and Karan in 2014 titled 'Proton Transport" 
"     Property in Supported Nafion Nanothin Films by Electrochemical Impedance" 
"     Spectroscopy' [1]"

# Note: Interpolations can only be done within the ranges given by the above
#       mentioned paper [1]; therefore, this model only works for properties of 
#       Nafion RH, temperature, and thickness of 20-95 %, 30-60 C, and 4-300 nm
#       respectively. Values outside of these ranges with result in an unknown
#       value of ionic conductivity.

# Note: Another valuable paper for model inputs is "An improved two-dimensional 
#       agglomerate cathode model to study the influence of catalyst layer 
#       structural parameters" by Sun, Peppley, and Karan from 2005 [2]. This
#       includes GDL and CL thickness, porosities, and more.

# Note: Permeability values taken as scaled values from Zenyuk, Das, Weber 
#       paper that reported saturated values for GDL and CL at reported
#       porosities. Title "Understanding Impacts of Catalyst Layer Thickness
#       on Fuel Cell Performance via Mathematical Modeling" (2016) [3]. 

# Note: Oxygen diffusion through Nafion was taken from Sethuraman et al. paper 
#       and scaled by water volume fraction as a function of t_naf. Titled 
#       "Measuring Oxygen, Carbon Monoxide, Hydrogen Sulfide Diffusion 
#       Coefficients and Solubility in Nafion Membranes" (2009) [4].

# Note: Knowledge of O2 diffusion scaling with water volume fraction used to
#       scale values. Interpolations from DeCaluwe et al. taken to evaluate
#       water fraction as function of t_naf. Titled "Structure-property 
#       relationships at Nafion thin-film interfaces:..." (2018) [5]

""" Import needed modules """
"-----------------------------------------------------------------------------"
import numpy as np
import cantera as ct

""" User Input Parameters """
"-----------------------------------------------------------------------------"
ctifile = 'pemfc.cti'

" Initial electrochemical values "
i_OCV = 0                           # 0 for OCV [A/cm^2] -> single != 0 
i_ext0 = np.linspace(0.001,0.1,10)  # external currents close to 0 [A/cm^2]
i_ext1 = np.linspace(0.101,0.5,10)  # external currents [A/cm^2]
i_ext2 = np.linspace(0.51,0.83,10)  # external currents further from 0 [A/cm^2]
phi_ca_init = 0.7                   # initial cathode potential [V]
T_ca, P_ca = 323, ct.one_atm      # cathode temp [K] and pressure [Pa] at t = 0
T_an, P_an = 323, ct.one_atm      # anode temp [K] and pressure [Pa] at t = 0

" Transport and material properties "
RH = 95              # relative humidity of Nafion phase [%]
C_dl = 1.5e-9        # capacitance of double layer [F/m^2]

" Pt loading and geometric values "
area_calcs = 0      # control for area calculations [0:p_Pt, 1:Pt_loading]
p_Pt = 10           # percentage of Pt covering the carbon particle surface [%]
w_Pt = 0.4          # loading of Pt on carbon [mg/cm^2]
rho_Pt = 21.45e3    # density of Pt for use in area property calcs [kg/m^3]
r_c = 25e-9         # radius of single carbon particle [m]
r_Pt = 1e-9         # radius of Pt 1/2 sphere sitting on C surface [m]
t_naf = 25e-9       # thickness of nafion around a carbon particle [m]
t_ca = 15e-6        # thickness of cathode CL modeled in simulation [m]
t_gdl = 250e-6      # thickness of cathode GDL modeled in simulation [m]
eps_gas = 0.1       # volume fraction of gas in CL [-]
eps_gdl = 0.5       # porosity of GDL [-]

" Gas channel boundary conditions "
Y_ca_BC = 'N2: 0.79, O2: 0.21, Ar: 0.01, H2O: 0.03, OH: 0'
T_ca_BC, P_ca_BC = 323, ct.one_atm  # cathode gas channel T [K] and P [Pa]

" Reaction properties "
n_elec_an = 2       # sum(nu_k*z_k) for anode surface reaction from ctifile

" Model Inputs "
ind_e = 0           # index of electrons in Pt surface phase... from cti
method = 'BDF'      # method for solve_ivp [eg: BDF,RK45,LSODA,Radau,etc...]
t_sim = 1e2         # time span of integration [s]
Ny_gdl = 3          # number of depth discretizations for GDL
Ny = 5              # number of depth discretizations for CL
Nr = 5              # number of radial discretizations for CL nafion shells

" Modify Tolerance for convergence "
max_t = t_sim       # maximum allowable time step for solver [s]
atol = 1e-9         # absolute tolerance passed to solver
rtol = 1e-6         # relative tolerance passed to solver

" Plot toggles - ('0' off and '1' on) "
post_only = 0       # turn on to only run post-processing plots
debug = 0           # turn on to plot first node variables vs time
radial = 0          # turn on radial O2 plots for each Nafion shell
grads = 0           # turn on to plot O2 and Phi gradients in depth of cathode
polar = 1           # turn on to generate full cell polarization curves
over_p = 0          # turn on to plot overpotential curve for cathode side

" Saving options "
folder_name = 'folder_name'  # folder name for saving all files/outputs
save = 0                     # toggle saving on/off with '1' or '0'

""" Process inputs from this file and run model """
"-----------------------------------------------------------------------------"
if __name__ == '__main__':
    exec(open("particle_shell_pemfc_pre.py").read())
    
    if post_only == 0:
        exec(open("particle_shell_pemfc_dsvdt.py").read())
        
    exec(open("particle_shell_pemfc_post.py").read())