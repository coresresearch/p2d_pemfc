"""
Butler-Volmer Test for Core-Shell Model
"""

import cantera as ct
import numpy as np
import pylab as plt

ctifile = 'pemfc.cti'
gas_ca = ct.Solution(ctifile, 'cathode_gas')
carb_ca = ct.Solution(ctifile, 'metal')
naf_b_ca = ct.Solution(ctifile, 'naf_bulk_ca')
naf_s_ca = ct.Interface(ctifile, 'naf_surf_ca', [naf_b_ca, gas_ca])
pt_s_ca = ct.Interface(ctifile, 'Pt_surf_ca', [carb_ca, naf_b_ca])

n_elec = 4
delta_phi_eq = -pt_s_ca.delta_gibbs / (n_elec*ct.faraday)

carb_ca.electric_potential = 0
pt_s_ca.electric_potential = 0

k_fwd = pt_s_ca.forward_rate_constants[0]
k_rev = pt_s_ca.reverse_rate_constants[0]

beta = 0.5
RT = ct.gas_constant*gas_ca.T

k_fwd_star = k_fwd / np.exp(-beta*n_elec*ct.faraday*delta_phi_eq / RT)
k_rev_star = k_rev / np.exp((1-beta)*n_elec*ct.faraday*delta_phi_eq / RT)

nu_fwd = pt_s_ca.reactant_stoich_coeffs().T
nu_rev = pt_s_ca.product_stoich_coeffs().T

Concs = np.hstack([carb_ca.concentrations, naf_b_ca.concentrations, pt_s_ca.concentrations])

i_o = n_elec*ct.faraday*k_fwd_star**(1-beta)*k_rev_star**beta \
    *np.prod(Concs**((1-beta)*nu_fwd))*np.prod(Concs**(beta*nu_rev))
    
Eta = np.linspace(0,0.3,50)
i_BV = np.zeros_like(Eta)

for ind,E in enumerate(Eta):
    i_BV[ind] = i_o*(np.exp(-beta*n_elec*ct.faraday*E/RT)
                     - np.exp((1-beta)*n_elec*ct.faraday*E/RT))
    
plt.plot(Eta,i_BV)
plt.xlabel('Overpotential [V]', family='Times New Roman')
plt.ylabel('Current [A / m^2 Pt]', family='Times New Roman')