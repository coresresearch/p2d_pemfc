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
"     and thickness and can be modeled via various sub models described in"
"     detail in the pemfc_property_func file."

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

# Note: Nafion conducitivity treatment as bulk material was added as an option.
#       This method assumes the thin shell has the same conductivity as bulk
#       material and that there are no confinement or microstructure effects
#       when it gets too thin. The relationship is taken from Yadav et al. in
#       their 2012 publication of "Analysis of EIS Technique and Nafion 117 
#       Conductivity as a Function of Temperature and Relative Humidity" [6]

# Note: Low Pt loading data and modeling results are present with operating 
#       conditions in "Modeling and Experimental Validation of Pt Loading and
#       Electrode Composition Efects in PEM Fuel Cells" by L. Hao et. al. [7]
#       This paper was published in 2015 and makes it recent enough to validate
#       against.

# Note: In order to validate the model, results for the full cell are needed.
#       The anode was added by determining the steady-state potential via
#       Gibb's free energy correlations. The PEM was simply modeled as a V=IR
#       relationship where the resistance (Ohm*cm^2) was taken from [8] (2002).

""" Import needed modules """
"-----------------------------------------------------------------------------"
import numpy as np
import cantera as ct

""" User Input Parameters """
"-----------------------------------------------------------------------------"
ctifile = 'pemfc.cti'

" Initial electrochemical values "
i_OCV = 0                           # 0 [A/cm^2] -> or single if polar == 0 
i_ext0 = np.linspace(0.001,0.1,10)  # external currents close to 0 [A/cm^2]
i_ext1 = np.linspace(0.101,3.6,10)  # external currents [A/cm^2]
i_ext2 = np.linspace(3.7,3.8,20)    # external currents further from 0 [A/cm^2]
phi_ca_init = 0.7                   # initial cathode potential [V]
T_ca, P_ca = 333, 1.5*ct.one_atm    # cathode temp [K] and pressure [Pa] at t = 0
T_an, P_an = 333, 1.5*ct.one_atm    # anode temp [K] and pressure [Pa] at t = 0

" Transport and material properties "
RH = 95              # relative humidity of Nafion phase [%]
C_dl = 1.5e-9        # capacitance of double layer [F/m^2]
sig_method = 'bulk'  # 'lam', 'bulk', or 'mix' for conductivity method
D_O2_method = 'bulk' # 'lam', 'bulk', or 'mix' for D_O2 method
R_naf = 45e-3        # resistance of Nafion membrane [Ohm*cm^2]

" Pt loading and geometric values "
area_calcs = 1      # control for area calculations [0:p_Pt, 1:Pt_loading]
p_Pt = 10           # percentage of Pt covering the carbon particle surface [%]
w_Pt = 0.2          # loading of Pt on carbon [mg/cm^2]
rho_Pt = 21.45e3    # density of Pt for use in area property calcs [kg/m^3]
r_c = 50e-9         # radius of single carbon particle [m]
r_Pt = 1e-9         # radius of Pt 1/2 sphere sitting on C surface [m]
t_naf = 5e-9        # thickness of nafion around a carbon particle [m]
t_cl = 15e-6        # thickness of cathode CL modeled in simulation [m] 
t_gdl = 250e-6      # thickness of cathode GDL modeled in simulation [m]
eps_gas = 0.1       # volume fraction of gas in CL [-]
eps_gdl = 0.5       # porosity of GDL [-]
theta = 45          # max transport angle (<90) for Nafion SA assuming planar

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
Ny = 5              # number of depth discretizations for CL
Nr = 5              # number of radial discretizations for CL nafion shells

" Modify Tolerance for convergence "
max_t = t_sim       # maximum allowable time step for solver [s]
atol = 1e-9         # absolute tolerance passed to solver
rtol = 1e-6         # relative tolerance passed to solver

" Plot toggles - (0: off and 1: on) "
post_only = 0       # turn on to only run post-processing plots
debug = 0           # turn on to plot first node variables vs time
radial = 0          # turn on radial O2 plots for each Nafion shell
grads = 0           # turn on to plot O2 and Phi gradients in depth of cathode
polar = 1           # turn on to generate full cell polarization curves
over_p = 0          # turn on to plot overpotential curve for cathode side

" Saving options "
folder_name = 'bulk_5nm'     # folder name for saving all files/outputs
save = 1                     # toggle saving on/off with '1' or '0'

" Font controls on axes, legends, etc "
ft = 14

#R_naf = (-28571*w_Pt**3 + 11667*w_Pt**2 - 1550*w_Pt + 111.9)*1e-3

""" End of user inputs - do not edit anything below this line """
"-----------------------------------------------------------------------------"
###############################################################################
###############################################################################
###############################################################################

""" Process inputs from this file and run model """
"-----------------------------------------------------------------------------"
if __name__ == '__main__':       
    exec(open("particle_shell_pemfc_pre.py").read())
    
    if post_only == 0:
        exec(open("particle_shell_pemfc_dsvdt.py").read())
        
    exec(open("particle_shell_pemfc_post.py").read())
