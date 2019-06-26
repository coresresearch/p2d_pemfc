"""
Created on Wed Jun 19 11:13:05 2019

Author: Corey R. Randall

Optimization routine:
    Minimize the error between the data from Owejan et. al. paper and the PEMFC
    models (core-shell and flooded-agglomerate) by varying the following:
    theta, offset, R_naf, and i_o. 
    
Instructions:
    To use this script, open up the pemfc_runner.py file and comment out the 
    line of code that specifies w_Pt. Copy and paste the lines of code
    below to input i_OCV, i_ext1, i_ext2, and i_ext3 into pemfc_runner.py. Once
    these items have been complete, fill out the initial guess values for the 
    optimization in 'p0', 'p1', or 'p2' in the order commented below. Ranges 
    for each of these parameters should also be specified in the respective 
    'bounds*' variable below. Simply run this script after these inputs have 
    been filled out and allow for the error to be minimized for the data sets 
    using the package scipy.optimize.minize.
    
Optimization options:
    The 'tog' variable can be set to 0, 1, or 2 in order to change which set of
    parameters are being varied in the optmization. When tog = 0 then R_naf, 
    theta, i_o, and offset are all varied and assumed to be constant accross all
    Pt loadings. If tog = 1 then the same parameters are varied except i_o is
    treated as a function of Pt loading as i_o = a*w_Pt + b which allows both a
    and b to be varied instead of just a constant i_o. The last case is when 
    tog = 2 which again assumes the same parameters are being varied except that
    R_naf is treated as a function of Pt loading instead of i_o. The form of 
    this function is R_naf = (c*w_Pt^d +35)*1e-3 where c and d become the fitting 
    parameters.
    
Units for inputs:
    R_naf [Ohm*cm^2], theta [degrees], i_o [A/cm^2 *(cm^3/kmol)^6.5], offset [-], 
    a [A/cm^2 *(cm^3/kmol)^6.5 /(mg/cm^2)], b [A/cm^2 *(cm^3/kmol)^6.5], 
    c [mOhm*cm^2 /(mg/cm^2)], and d [-].
"""

import numpy as np
import cantera as ct

# How to set up i_ext vector in the pemfc_runner.py file:
"""
i_OCV = 0
i_ext1 = np.array([0.05, 0.20, 0.40, 0.80, 1.0, 1.2, 1.5, 1.65, 1.85, 2.0])
i_ext2 = np.array([])
i_ext3 = np.array([])"""

# User Inputs for optimization of R_naf, theta, i_o, and offset:
p0 = np.array([45e-3, 45, 4e-6, 0.55]) 
bounds0 = np.array([[20e-3, 120e-3], [42.5, 80], [1e-6, 9e-6], [0.52, 1.0]]) # [min, max]

# User Inputs for optimization of R_naf, theta, offset, a, and b from i_o = a*w_Pt + b:
p1 = np.array([45e-3, 45, 0.55, 24.5e-6, -0.60e-6])
p1 = np.array([54.6322292e-3, 44.9866196, 0.545933492, 24.18e-6, -0.441002786e-6])
bounds1 = np.array([[20e-3, 120e-3], [42.5, 80], [0.52, 1.0], [24.17e-6, 50e-6], [-0.5e-6, 1e-6]])

# User Inputs for optimization of theta, i_o, offset, c, and d from R_naf = (c*w_Pt^d +35)*1e-3:
p2 = np.array([45, 2e-6, 0.55, 0.874, -1.083])
#p2 = np.array([50, 1.58523583e-6, 0.535449905, 0.995444795, -1.16389853])
bounds2 = np.array([[42.5, 80], [5e-7, 9e-6], [0.52, 1.0], [0., 1.5], [-1.5, 0.]])

# Toggle for which optimization to run (0, 1, or 2):
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
y_data = np.hstack([y1[1:], y2[1:], y3[1:], y4[1:]])
s_data = np.hstack([s1[1:], s2[1:], s3[1:], s4[1:]])

def chi_sq_func0(p_opt): # p_opt[:] = R_naf, theta, i_o, offset
    
    global w_Pt, R_naf_opt, theta_opt, i_o_opt, offset_opt
    R_naf_opt, theta_opt, i_o_opt, offset_opt = p_opt
    print('\n\nR_naf, theta, i_o, offset:', p_opt)
    
    y_model = np.array([])
    w_Pt_vec = np.array([0.2, 0.1, 0.05, 0.025])
    for i, w in enumerate(w_Pt_vec):
        w_Pt = w
        exec(open("pemfc_runner.py").read(), globals(), globals())
        y_model = np.hstack([y_model, dphi_ss[1:]])
        
    chi_sq = sum((y_data - y_model)**2 / s_data**2)
    print('\nchi_sq:', chi_sq)
    
    return chi_sq

# Results for by-hand fit: (comparison to tog = 0)
"""
chi_sq = 884.6912413825394

R_naf = user_inputs.R_naf = 45e-3
theta = user_inputs.theta = 45.
i_o = 4e-6
offset = 0.55"""

# Results for tog = 0:
"""
chi_sq = 442.3504466112417

R_naf = user_inputs.R_naf = 47.0708165e-3
theta = user_inputs.theta = 44.9310232
i_o = 1.47659887e-6
offset = 0.549173549"""

def chi_sq_func1(p_opt): # p_opt[:] = R_naf, theta, offset, a, b
    w_Pt_vec = np.array([0.2, 0.1, 0.05, 0.025])
    
    global w_Pt, R_naf_opt, theta_opt, offset_opt, a, b
    R_naf_opt, theta_opt, offset_opt, a, b = p_opt
    print('\n\nR_naf, theta, offset, a, b:', p_opt)
    print('i_o:', abs(a*w_Pt_vec + b))
    
    y_model = np.array([])
    for i, w in enumerate(w_Pt_vec):
        w_Pt = w
        exec(open("pemfc_runner.py").read(), globals(), globals())
        y_model = np.hstack([y_model, dphi_ss[1:]])
        
    chi_sq = sum((y_data - y_model)**2 / s_data**2)
    print('\nchi_sq:', chi_sq)
    
    return chi_sq

# Results for by-hand with varying i_o: (comparison to tog = 1)
"""
chi_sq = 403.90458836954605

R_naf = user_inputs.R_naf = 45e-3
theta = user_inputs.theta = 45.
offset = 0.55
a = 24.5e-6
b = -0.60e-6"""
    
# Results for tog = 1:
"""
chi_sq = 111.3265875677507

R_naf = user_inputs.R_naf = 54.4607050e-3
theta = user_inputs.theta = 44.9763477
offset = 0.545805050
a = 24.17e-6
b = -0.434032963e-6"""

def chi_sq_func2(p_opt): # p_opt[:] = theta, i_o, offset, c, d
    w_Pt_vec = np.array([0.2, 0.1, 0.05, 0.025])
    
    global w_Pt, theta_opt, i_o_opt, offset_opt, c, d
    theta_opt, i_o_opt, offset_opt, c, d = p_opt
    print('\n\ntheta, i_o, offset, c, d:', p_opt)
    print('R_naf:', (c*w_Pt_vec**d + 35) *1e-3)
    
    y_model = np.array([])
    for i, w in enumerate(w_Pt_vec):
        w_Pt = w
        exec(open("pemfc_runner.py").read(), globals(), globals())
        y_model = np.hstack([y_model, dphi_ss[1:]])
        
    chi_sq = sum((y_data - y_model)**2 / s_data**2)
    print('\nchi_sq:', chi_sq)
    
    return chi_sq

# Results for by-hand with varying R_naf: (comparison to tog = 2)
"""
chi_sq = 203.66835835935382

theta = user_inputs.theta = 45.
i_o = 2e-6
offset = 0.55
c = 0.874
d = -1.083"""
    
# Results for tog = 2:
"""
chi_sq = 69.49687105275932

theta = user_inputs.theta = 50.2228419
i_o = 1.27966523e-6
offset = 0.653522006
c = 0.929923592
d = -1.24932356"""

if tog == 0:
    res = minimize(chi_sq_func0, p0, method='L-BFGS-B', bounds=bounds0)
elif tog == 1:
    res = minimize(chi_sq_func1, p1, method='L-BFGS-B', bounds=bounds1)
elif tog == 2:
    res = minimize(chi_sq_func2, p2, method='L-BFGS-B', bounds=bounds2)
    
# Use this in w_Pt_runner to see what plots look like:
""" Post this line of code in w_Pt_runner.py between line 38 and line 40. Set
the tog value to the correct option (0, 1, or 2) based on what parameters are
being varied. Then set the needed values equal to their optimal parameters while
commenting out any that are unneeded. Run w_Pt_runner.py once done to see the
results.

optimize, tog = 1, 1
R_naf_opt = user_inputs.R_naf = 54.4607050e-3
theta_opt = user_inputs.theta = 44.9763477
i_o_opt = 1.27966523e-6
offset_opt = 0.545805050
a = 24.17e-6
b = -0.434032963e-6
c = 0.929923592
d = -1.24932356"""
    
    