"""
Database for lamellae samples from DeCaluwe [5] compared to Karan [1]. This 
code generates scaling factors used in a series resistor model for calculating
O2 diffusion coefficients through Nafion shells assuming that lamellae exist.
"""

" Load useful modules "
import copy, math
import numpy as np
from scipy.optimize import minimize


" Constants "
SLD_naf = 4.15743
SLD_wat = -0.559883
rho_naf = 1.98
rho_wat = 1.006
MW_wat =18.016
EW = 1100


" Data from DeCaluwe Nano Energy Paper "
# t values represent thicknesses from paper, values in parenteses represent the
# total hydrated thickness found from summing over all lamellae at 92% RH. 
# Karan values represent the thicknesses from [1] that were compared to closest
# t values examined in [5].
SLDs = np.array([[-0.2237, 4.5043, 1.6979, 3.9113, 4.15743, 4.15743, 4.15743], #t5 (5.9) -> Karan4
                [-0.0958, 4.373, 2.583, 3.791, 2.6508, 4.15743, 4.15743], #t7 (8.9) -> Karan10
                [0.8616, 4.3067, 2.8787, 3.7708, 3.2241, 3.5187, 3.3272], #t42 (51) -> Karan55
                [-0.3722, 3.5109, 1.4651, 3.2542, 2.5756, 2.9446, 3.1720]]) #t120 (162) -> Karan160

t_i = np.array([[8.612, 24.118, 10.505, 15.281, 0, 0, 0], #t5 (5.9) -> Karan4
                [7.5947, 20.907, 15.228, 18.3322, 26.467, 0, 0], #t7 (8.9) -> Karan10
                [10.866, 16.236, 16.600, 16.202, 21.114, 11.830, 417.886], #t42 (51)-> Karan55
                [9.481, 22.070, 16.776, 19.297, 18.831, 1477.106, 56.649]]) #t120 (162) -> Karan160


" Calculations to get V_w for each lamellae "
V_w = (SLD_naf - SLDs) / (SLD_naf - SLD_wat)

V_w_low = np.array([0.659, 0.108]) # from DeCaluwe Soft Matter Paper - t5 fit
Lambda_calc_low = np.array([34.6, 7.1]) # from DeCaluwe Soft Matter Paper - t5 fit

SLD_high = np.array([0.32, 3.78]) # from DeCaluwe Nano Energy Paper - profile averages
V_w_high = (SLD_naf - SLD_high) / (SLD_naf - SLD_wat)
Lambda_calc_high = rho_wat*EW*V_w_high /(1-V_w_high) /MW_wat /rho_naf

V_w[0:3,0:2] = V_w_low
V_w[3:,0:2] = V_w_high


" Pre-processing for conductivity fitting model "
Lambda_i = rho_wat*EW*V_w /(1-V_w) /MW_wat /rho_naf
Lambda_i[0:3,0:2] = Lambda_calc_low[0:2]
Lambda_i[3:,0:2] = Lambda_calc_high[0:2]

T = 302.75
sigma_30_Karan = np.array([21.481, 22.062, 20.185, 34.810, 43.913])/10. # Karan data
sigma_io_i = math.exp(1286.*(1./303. - 1./T))*(0.514*Lambda_i - 0.326) # Ref ??


" Model to fit Karan conductivities - produces fitting parameters "
C3_1 = 0.333
C3_2 = 0.754
C3_3 = 0.671

C0 = np.array([C3_1, C3_2, C3_3])

def RSS_func(C,flag=0.0):

    Clam1_M3 = C[0]
    Clam2_M3 = C[1]
    Cfilm_M3 = C[2]  
    
    t_i_norm = copy.deepcopy(t_i)
    t_i_norm[t_i_norm==0] = 1.0
    
    sigma_io_i_M3 = copy.deepcopy(sigma_io_i)

    sigma_io_i_M3 *= Cfilm_M3
    sigma_io_i_M3[:,0] *= ((1.0 - 0./4.)*Clam1_M3 + (0./4.)*Clam2_M3)
    sigma_io_i_M3[:,1] *= ((1.0 - 1./4.)*Clam1_M3 + (1./4.)*Clam2_M3)
    sigma_io_i_M3[:,2] *= ((1.0 - 2./4.)*Clam1_M3 + (2./4.)*Clam2_M3)
    sigma_io_i_M3[:,3] *= ((1.0 - 3./4.)*Clam1_M3 + (3./4.)*Clam2_M3)
    sigma_io_i_M3[:,4] *= ((1.0 - 4./4.)*Clam1_M3 + (4./4.)*Clam2_M3)

    R_io_trans_eff_M3 = np.zeros(t_i.shape[0])
    sigma_io_trans_eff_M3 = np.zeros(t_i.shape[0])
    sigma_io_par_eff_M3 = np.zeros(t_i.shape[0])
    t_film = np.zeros(t_i.shape[0])

    for j in np.arange(t_i.shape[0]):
        t_film[j] = sum(t_i[j,:])
        for i in np.arange(t_i.shape[1]):
            sigma_io_par_eff_M3[j] += sigma_io_i_M3[j,i]*t_i[j,i] / t_film[j]
            R_io_trans_eff_M3[j] += t_i[j,i] / sigma_io_i_M3[j,i] / t_film[j]

    sigma_io_trans_eff_M3 = 1 / R_io_trans_eff_M3
    
    RSS_3 = (sigma_io_par_eff_M3[0] - sigma_30_Karan[0])**2 \
          + (sigma_io_par_eff_M3[1] - sigma_30_Karan[1])**2 \
          + (sigma_io_par_eff_M3[2] - sigma_30_Karan[2])**2 \
          + (sigma_io_par_eff_M3[3] - sigma_30_Karan[3])**2

    if flag==0:
        return RSS_3
    elif flag == 1:
        return t_film
    elif flag ==2:
        return sigma_io_par_eff_M3
    elif flag == 3:
        return sigma_io_trans_eff_M3

bounds = np.array([[0.0,1.0],[0.0,1.0],[0.0,1.0]])
res = minimize(RSS_func, C0, method='L-BFGS-B', bounds=bounds, args=(0.0,))
print( res.x )

" Bulk-like Data from DeCaluwe Nano Energy Paper, [5] "
# Take the thicknesses of the bulk-like layers from the lamellae results in
# order to see the relationship between water content and bulk-like thickness.
# User these results to scale sig_io and D_O2 by the water volume fractions so
# that these transport properties change as a function of Nafion thickness w/o
# requiring an assumption that lamellae exist in the electrolyte structure.

