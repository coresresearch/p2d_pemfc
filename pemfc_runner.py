""" Model Description """
"-----------------------------------------------------------------------------"
"""This model is a half cell model of of PEM fuel cell cathode. The runner file
allows the user to execute two different geometries for the catalyst layer:
core-shell and flodded-agglomerate. In the core-shell model, a Pt-covered 
carbon core is covered with a thin shell of Nafion. In the flooded-agglomerate,
multiple core-shell geometries are clustered together and wrapped in an another
shell of Nafion."""

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
import os, sys
import numpy as np
import cantera as ct
ct.add_directory(os.getcwd() + '/Core_Shell')
ct.add_directory(os.getcwd() + '/Flooded_Agg')

""" User Input Parameters """
"-----------------------------------------------------------------------------"
model = 'flooded_agg'               # CL geom: 'core_shell' or 'flooded_agg'
ctifile = 'pemfc_fa.cti'            # cantera input file to match chosen model
ver = 2                             # debugging radial diffusion (1:cs, 1-2:fa)

" Initial electrochemical values "
i_OCV = 0                           # 0 [A/cm^2] -> or single run if != 0 
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
R_naf = 60e-3        # resistance of Nafion membrane [Ohm*cm^2] (45:cs, 60:fa)

" Pt loading and geometric values "
area_calcs = 1      # control for area calculations [0:p_Pt, 1:Pt_loading]
p_Pt = 10           # percentage of Pt covering the carbon particle surface [%]
w_Pt = 0.2          # loading of Pt on carbon [mg/cm^2]
rho_Pt = 21.45e3    # density of Pt for use in area property calcs [kg/m^3]
r_c = 25e-9         # radius of single carbon particle [m] (50:cs, 25:fa)
r_Pt = 1e-9         # radius of Pt 1/2 sphere sitting on C surface [m]
t_gdl = 250e-6      # thickness of cathode GDL modeled in simulation [m]
t_cl = 15e-6        # thickness of cathode CL modeled in simulation [m]
t_naf = 5e-9        # thickness of nafion around each core/agglomerate [m]
eps_g_gdl = 0.5     # porosity of GDL [-]
eps_g_cl = 0.1      # porosity of CL [-]

" Core-shell specific geometry values "
theta = 45          # O2 transport angle (<90) for Nafion SA [degrees]

" Flooded-agglomerate specific geometry values "
p_c = 95            # % of C sheres packed in agglomerate - Gauss max = 74%
r_agg = 50e-9       # radius of agglomerate excluding nafion shell [m]

" Gas channel boundary conditions "
Y_ca_BC = 'N2: 0.79, O2: 0.21, Ar: 0.01, H2O: 0.03, OH: 0'
T_ca_BC, P_ca_BC = 333, 1.5*ct.one_atm  # cathode gas channel T [K] and P [Pa]

" Reaction properties "
n_elec_an = 2       # sum(nu_k*z_k) for anode surface reaction from ctifile

" Model inputs "
ind_e = 0           # index of electrons in Pt surface phase... from cti
method = 'BDF'      # method for solve_ivp [eg: BDF,RK45,LSODA,Radau,etc...]
t_sim = 1e2         # time span of integration [s]
Ny_gdl = 3          # number of depth discretizations for GDL
Ny_cl = 5           # number of depth discretizations for CL
Nr_cl = 5           # number of radial discretizations for CL nafion shells

" Modify tolerance for convergence "
max_t = t_sim       # maximum allowable time step for solver [s]
atol = 1e-9         # absolute tolerance passed to solver
rtol = 1e-6         # relative tolerance passed to solver

" Plot toggles - (0: off and 1: on) "
post_only = 0       # turn on to only run post-processing
debug = 1           # turn on to plot first node variables vs time
radial = 1          # turn on radial O2 plots for each Nafion shell
grads = 1           # turn on to plot O2 and Phi gradients in depth of cathode
polar = 1           # turn on to generate full cell polarization curves
over_p = 1          # turn on to plot overpotential curve for cathode side

" Verification settings "
i_ver = 1           # verify current between GDL and CL with O2 flux calcs
i_find = 0.5        # current from polarization curve to use in i_ver processing

" Plotting options "
font_nm = 'Arial'   # name of font for plots
font_sz = 14        # size of font for plots

" Saving options "
folder_name = 'cs_test_06122019'     # folder name for saving all files/outputs
save = 1                        # toggle saving on/off with '1' or '0'

""" End of user inputs - do not edit anything below this line """
"-----------------------------------------------------------------------------"
###############################################################################
###############################################################################
###############################################################################

""" Process inputs from this file and run model """
"-----------------------------------------------------------------------------"
if __name__ == '__main__':
    exec(open("Shared_Funcs/pemfc_pre.py").read())
    
    if post_only == 0:
        exec(open("Shared_Funcs/pemfc_dsvdt.py").read())
    
    exec(open("Shared_Funcs/pemfc_post.py").read())
