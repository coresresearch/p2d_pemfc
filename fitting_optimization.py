"""
Created on Wed Jun 19 11:13:05 2019

Author: Corey R. Randall

Optimization routine:
    Minimize the error between the data from Owejan et. al. paper and the PEMFC
    models (core-shell and flooded-agglomerate) by varying the following:
    theta, offset, R_naf, and i_o. 
    
Instructions:
    To use this script, open up the pemfc_runner.py file and comment out the 
    lines of code that specify R_naf, w_Pt, and theta. Use the lines of code
    below to input i_OCV, i_ext1, i_ext2, and i_ext3 into pemfc_runner.py. Once
    these items have been complete, fill out the initial guess values for the 
    optimization in 'p0' in the order commented below. Ranges for each of these
    parameters should also be specified in the 'bounds' variable below. Simply
    run this script after these inputs have been filled out and allow for the 
    error to be minimized for the data sets using scipy.optimize.minize.
"""

import numpy as np
import cantera as ct

# How to set up i_ext vector:
"""
i_OCV = 0
i_ext1 = np.array([0.05, 0.20, 0.40, 0.80, 1.0, 1.2, 1.5, 1.65, 1.85, 2.0])
i_ext2 = np.array([])
i_ext3 = np.array([])"""

# User Inputs for optimization of R_naf, theta, i_o, and offset:
p0 = np.array([45e-3, 50, 4e-6, 0.5]) # R_naf [Ohm*m^2], theta [degrees], i_o, offset
bounds0 = np.array([[20e-3, 120e-3], [50, 80], [1e-6, 9e-6], [0.3, 1.0]]) # [min, max]

# User Inputs for optimization of R_naf, theta, offset, a, and b from i_o = a*w_Pt + b:
p1 = np.array([45e-3, 50, 0.5, 23e-6, -0.6e-6])
bounds1 = np.array([[20e-3, 120e-3], [50, 80], [0.3, 1.0], [1e-6, 50e-6], [-1e-6, 1e-6]])

# User Inputs for optimization of theta, i_o, offset, c, and d from R_naf = c*w_Pt^d:
p2 = np.array([])
bounds = np.array([])

# Toggle for which optimization to run:
tog = 1

""" Do not edit anything below this line """
"-----------------------------------------------------------------------------"
###############################################################################
###############################################################################
###############################################################################

# Data to fit for different Pt loadings (x,y) -> (i [A/cm^2], Phi [V]):
" All x values are shared and y* is the y data, s* is the error bars (+/-) "
x = np.array([0.0, 0.05, 0.20, 0.40, 0.80, 1.0, 1.2, 1.5, 1.65, 1.85, 2.0])
y1 = np.array([0.95, 0.85, 0.80, 0.77, 0.73, 0.72, 0.70, 0.68, 0.67, 0.65, 0.63]) # w_Pt = 0.2 mg/cm^2
s1 = np.array([0.1, 12, 7, 7, 12, 1, 8, 7, 7, 9, 9]) *1e-3 

y2 = np.array([0.93, 0.83, 0.79, 0.75, 0.71, 0.69, 0.67, 0.65, 0.64, 0.62, 0.60]) # w_Pt = 0.1 mg/cm^2
s2 = np.array([0.1, 9, 7, 5, 7, 11, 11, 7, 9, 11, 11]) *1e-3

y3 = np.array([0.92, 0.81, 0.76, 0.72, 0.67, 0.65, 0.63, 0.60, 0.59, 0.56, 0.54]) # w_Pt = 0.05 mg/cm^2
s3 = np.array([0.1, 8, 6, 6, 7, 7, 5, 5, 6, 7, 7]) *1e-3

y4 = np.array([0.91, 0.79, 0.72, 0.68, 0.63, 0.60, 0.57, 0.53, 0.50, 0.46, 0.43]) # w_Pt = 0.025 mg/cm^2
s4 = np.array([0.1, 4, 10, 14, 13, 13, 19, 24, 25, 23, 24]) *1e-3

# Scipy optimization (try now) vs scikits-learn (later??):
optimize = 1 # tell pre processor to reset optimized parameters

from scipy.optimize import minimize
y_data = np.hstack([y1, y2, y3, y4])
s_data = np.hstack([s1, s2, s3, s4])

def chi_sq_func0(p_opt): # p_opt[:] = R_naf, theta, i_o, offset
    
    global w_Pt, R_naf_opt, theta_opt, i_o_opt, offset_opt
    R_naf_opt, theta_opt, i_o_opt, offset_opt = p_opt
    print('\n\nR_naf, theta, i_o, offset:', p_opt)
    
    y_model = np.array([])
    w_Pt_vec = np.array([0.2, 0.1, 0.05, 0.025])
    for i, w in enumerate(w_Pt_vec):
        w_Pt = w
        exec(open("pemfc_runner.py").read(), globals(), globals())
        y_model = np.hstack([y_model, dphi_ss])
        
    chi_sq = sum((y_data - y_model)**2 / s_data**2)
    print('\nchi_sq:', chi_sq)
    
    return chi_sq

# Results for by-hand fit: (comparison to tog = 0)
"""
chi_sq = 9148406.997429071

R_naf = user_inputs.R_naf = 
theta = user_inputs.theta = 45
i_o = 4e-6
offset = 0.55"""

# Results for tog = 0:
"""
chi_sq = 8551703.012265678

R_naf = user_inputs.R_naf = 78.3872409e-3
theta = user_inputs.theta = 62.4837816
i_o = 5.09982691e-6
offset = 1.0"""

def chi_sq_func1(p_opt): # p_opt[:] = R_naf, theta, offset, a, b
    w_Pt_vec = np.array([0.2, 0.1, 0.05, 0.025])
    
    global w_Pt, R_naf_opt, theta_opt, offset_opt, a, b
    R_naf_opt, theta_opt, offset_opt, a, b = p_opt
    print('\n\nR_naf, theta, offset, a, b:', p_opt, 'i_o:', abs(a*w_Pt_vec + b))
    
    y_model = np.array([])
    for i, w in enumerate(w_Pt_vec):
        w_Pt = w
        exec(open("pemfc_runner.py").read(), globals(), globals())
        y_model = np.hstack([y_model, dphi_ss])
        
    chi_sq = sum((y_data - y_model)**2 / s_data**2)
    print('\nchi_sq:', chi_sq)
    
    return chi_sq

# Results for by-hand with varying i_o: (comparison to tog = 1)
"""
chi_sq = 

R_naf = user_inputs.R_naf = 
theta = user_inputs.theta = 
offset = 
a = 
b = """
    
# Results for tog = 1:
"""
chi_sq = 

R_naf = user_inputs.R_naf = 
theta = user_inputs.theta =
offset = 
a = 
b = """

def chi_sq_func2(p_opt): # p_opt[:] = theta, offset, i_o, c, d, e
    w_Pt_vec = np.array([0.2, 0.1, 0.05, 0.025])
    
    global w_Pt, theta_opt, offset_opt, i_o_opt, c, d
    theta_opt, offset_opt, i_o_opt, c, d = p_opt
    print('\n\ntheta, offset, i_o, c, d:', p_opt, 'R_naf:', c*w_Pt_vec**d)
    
    y_model = np.array([])
    for i, w in enumerate(w_Pt_vec):
        w_Pt = w
        exec(open("pemfc_runner.py").read(), globals(), globals())
        y_model = np.hstack([y_model, dphi_ss])
        
    chi_sq = sum((y_data - y_model)**2 / s_data**2)
    print('\nchi_sq:', chi_sq)
    
    return chi_sq

# Results for by-hand with varying R_naf: (comparison to tog = 2)
"""
chi_sq = 

theta = user_inputs.theta = 
offset = 
i_o = 
c = 
d = """
    
# Results for tog = 2:
"""
chi_sq = 

theta = user_inputs.theta = 
offset = 
i_o = 
c = 
d = """

if tog == 0:
    res = minimize(chi_sq_func0, p0, method='L-BFGS-B', bounds=bounds0)
elif tog == 1:
    res = minimize(chi_sq_func1, p1, method='L-BFGS-B', bounds=bounds1)
elif tog == 2:
    res = minimize(chi_sq_func2, p2, method='L-BFGS-B', bounds=bounds2)
    
    