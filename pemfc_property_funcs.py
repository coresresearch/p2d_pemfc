"""
Calculate Nafion transport properties as a function of morphology/state:
    
    The Nafion electrolyte used in the particle_shell_pemfc_* files has a
    complex microstructure and morphology that affects the value of important
    parameters that are used in the model, i.e. ionic conductivity and 
    effective oxygen diffusion coefficient.
    
    These parameters are functions of the microstructure of Nafion as well as 
    the state of the local environment, i.e. temp, RH, etc. The defined 
    functions below take information found in relavent literature and 
    approximate appropriate transport parameters based on the supplied user 
    inputs from the model.    
"""



""" Import helpful modules """
"-----------------------------------------------------------------------------"
import numpy as np



""" Interpolate Nafion ionic conductivity [S/m] """
"-----------------------------------------------------------------------------"
def sig_naf_io_int(temp, t_naf, RH):
    
    # Data below is taken from "Proton Transport in Supported Nafion Nanothin
    # Films by Electrochemical Impedence Spectroscopy" by Paul, MacCreery, and
    # Karan in their Supporting Information Document [1]. The data was given in 
    # mS/cm and converted to S/m for the model calling this function.
    
    # indecies: temp(C), thickness(nm), RH(%) 
    sig_data = np.zeros([4,5,5]) 
    temp_vals = np.array([30,40,50,60])
    thickness_vals = np.array([4,10,55,160,300])
    RH_vals = np.array([20,40,60,80,95])
    
    "Data for 30C as thickness(nm) for rows and RH(%) for columns"
    sig_data[0,:,:] = np.array([[0.0001,0.012,0.278,3.432,21.481],
                                [0.0003,0.018,0.339,3.895,22.062],
                                [0.0004,0.028,0.550,4.296,20.185],
                                [0.0016,0.081,1.120,9.244,34.810],
                                [0.0071,0.359,2.797,10.978,43.913]])
    
    "Data for 40C as thickness(nm) for rows and RH(%) for columns"
    sig_data[1,:,:] = np.array([[0.0003,0.029,0.585,6.164,30.321],
                                [0.0009,0.034,0.625,5.374,48.799],
                                [0.0011,0.065,0.931,6.909,40.439],
                                [0.0032,0.152,1.770,14.162,68.326],
                                [0.0140,0.605,4.939,17.083,68.334]])
    
    "Data for 50C as thickness(nm) for rows and RH(%) for columns"
    sig_data[2,:,:] = np.array([[0.001,0.062,1.087,8.335,37.686],
                                [0.002,0.077,1.031,8.127,57.339],
                                [0.002,0.121,1.603,9.149,48.934],
                                [0.007,0.247,2.704,19.221,72.006],
                                [0.031,1.076,7.185,20.981,83.923]])
    
    "Data for 60C as thickness(nm) for rows and RH(%) for columns"
    sig_data[3,:,:] = np.array([[0.003,0.14,1.51,11.16,55.18],
                                [0.003,0.17,1.72,13.67,62.39],
                                [0.007,0.24,2.29,16.60,63.20],
                                [0.015,0.45,4.31,26.63,93.33],
                                [0.009,0.44,3.43,26.73,100.60]])
    
    "Create interpolation function for relavent ranges"
    from scipy.interpolate import RegularGridInterpolator
    sig_io_func = RegularGridInterpolator((temp_vals,thickness_vals,RH_vals),
                                           sig_data)
    
    "Call interpolation function for model specified paramaters"
    # Multiplication by 0.1 is unit conversion from mS/cm to S/m. Runner file
    # stores T and t_naf in [K] and [m] so are also converted inside the 
    # interpolation function to the same units as original data [C] and [nm].
    sig_naf_io = sig_io_func([temp-273, t_naf*1e9, RH])*0.1     
    
    # Output returns ionic conductivity [S/m]
    return sig_naf_io



""" Calculate effective O2 diffusion coeff. in Nafion """
"-----------------------------------------------------------------------------"
def D_eff_naf_int(temp, t_naf):
    
    # Relationships below use temperature to approximate an oxygen diffusion
    # coefficient in bulk Nafion from [4]. This value is then normalized by
    # the water volume from their samples and rescaled by the assumed water
    # volume in the shell based on t_naf and data taken from [5].
    
    # Inputs: temperature [K], Nafion shell thickness [m]
    
    "Use arrhenius relationship from Sethuraman et al. [4] to get ref D_O2"
    D_eff_ref = 17.45e-6*np.exp(-1514 /temp) /100**2 # 100**2 [cm^2/s -> m^2/s]
    
    "Scale by water volume fraction -> a function of t_naf from DeCaluwe [5]"
    tNaf_dat = np.array([4.6, 6.8, 11.6, 17.8, 19.5, 41.9, 59.9, 102.8, 119.6, 
                         154.0])*1e-9 # Nafion thicknesses from study
    
    tNb = tNaf_dat[2:] # Samples with mix of lamellae and bulk layers
    
    # Water volume fractions for: whole film average, bulk layers, lamellae
    V_wa = np.array([0.152, 0.165, 0.160, 0.181, 0.255, 0.254, 0.263, 0.259])
    V_wb = np.array([0.122, 0.152, 0.147, 0.176, 0.246, 0.246, 0.258, 0.255])
    V_wl = np.array([0.214, 0.235, 0.167, 0.184, 0.181, 0.201, 0.325, 0.407, 
                     0.374, 0.368])
    
    # Create interpolation vector of V_w values for whole range
    V_w_vec = np.hstack([V_wl[0:2], V_wa])
    V_w_interp = np.interp(t_naf, tNaf_dat, V_w_vec)
    
    # Normalize V_w with 0.37 - from lambda=18 in [4]
    D_eff_naf = D_eff_ref /0.37 *V_w_interp 
    
    # Output returns effective O2 diffusion coefficient [m^2/s]
    return D_eff_naf



"""    
Offload geometric calculations for reaction areas to clean up the code:
    
    Using user input for carbon and Pt radii as well as a Pt-loading (common
    in PEMFC fabrication), the reaction areas are estimated. These calculations
    were combined into this set of functions in order to provide a shorter code
    for the runner and dsvdt functions therefore making them more managable to
    edit and debug.
"""



""" Caclulate the reaction areas from Pt-loadding and radii information """
"-----------------------------------------------------------------------------"
def rxn_areas(w_Pt, t_ca, eps_gas, t_naf, r_c, r_Pt, rho_Pt):
    # Units for inputs are:
    # Pt_loading [mg/cm^2], A_LW_ca [cm^2], t_ca [m], eps_gas [-], t_naf [m], 
    # r_c [m], r_Pt [m], and rho_Pt [kg/m^3]
    
    "Calculate the surface area of the carbon particle and agglomerate volume"
    SA_C = 4*np.pi*r_c**2
    V_C_agg = 4/3*np.pi*r_c**3
    V_agg = 4/3*np.pi*(r_c + t_naf)**3
    N_agg = (1 - eps_gas)*100**2*t_ca / V_agg  # 100**2 [cm^2] -> [m^2]
    
    "Distribute total Pt mass by estimated number of agglomerates"
    W_Pt = w_Pt*0.001 # 0.001 converts mg to kg
    V_Pt = W_Pt / rho_Pt
    V_Pt_agg = V_Pt / N_agg
    V_Pt_bulb = 2/3*np.pi*r_Pt**3   # volume of Pt 1/2 sphere sitting on C surf
    N_bulbs = V_Pt_agg / V_Pt_bulb
    
    "Using r_Pt and assuming semi-spheres, find Pt surface area and volume"
    SA_Pt_agg = N_bulbs*2*np.pi*r_Pt**2
    SA_C_int = SA_C - N_bulbs*np.pi*r_Pt**2 + SA_Pt_agg
    SA_naf_int = 4*np.pi*(r_c + t_naf)**2
    V_C_Pt = V_C_agg + V_Pt_agg
    V_naf = V_agg - V_C_Pt
  
    # Units for outputs are:
    # SA_Pt_agg [m^2], SA_C_int [m^2], SA_naf_int [m^2], V_naf [m^3], and 
    # V_agg [m^3]
    return SA_Pt_agg, SA_C_int, SA_naf_int, V_naf, V_agg