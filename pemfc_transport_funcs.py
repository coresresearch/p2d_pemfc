""" 
Transport functions used in pemfc particle-shell model:
    Positive flux implies net flux from the TDY1 node into the TDY2 node. Make
    sure to correctly emphasize signs in runner codes calling this function."""
    
import numpy as np

def fickian_ADF(TDY1, TDY2, gas, params, toggle):
    # set state 1 properties:
    gas.TDY = TDY1
    D_k1 = gas.mix_diff_coeffs_mass
    rho1 = gas.density_mass
    mu1 = gas.viscosity
    P1 = gas.P
    Y1 = gas.Y
    rho_k1 = rho1*Y1
    
    # set state 2 properties:
    gas.TDY = TDY2
    D_k2 = gas.mix_diff_coeffs_mass
    rho2 = gas.density_mass
    mu2 = gas.viscosity
    P2 = gas.P
    Y2 = gas.Y
    rho_k2 = rho2*Y2
    
    # calculate average boundary properties:
    D_k = 0.5*(D_k1 + D_k2)
    rho = 0.5*(rho1 + rho2)
    mu = 0.5*(mu1 + mu2)
    rho_k = 0.5*(rho_k1 + rho_k2)
    
    # convective and diffusive driving terms:
    J_conv = -rho_k*params['K_g']*(P2 - P1)*params['1/dy'] / mu
    J_diff = -params['phi/tau_sq']*D_k*rho*(Y2 - Y1)*params['1/dy']
    
    # net mass flux of each species:
    mass_flux = J_conv + J_diff
    
    if toggle == 0:
        mass_flux = np.zeros(gas.n_species)
    
    return mass_flux