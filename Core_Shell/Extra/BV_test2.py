"""
Test BV for Core-Shell Model using kinetic fsolve approach...
"""

import cantera as ct
import numpy as np
from scipy.optimize import fsolve
import matplotlit.pyplot as plt

# cti input file:
ctifile = 'pemfc.cti'

# Create phases:
gas_an = ct.Solution(ctifile, 'gas_an')
gas_ca = ct.Solution(ctifile, 'gas_ca')
anode = ct.Solution(ctifile, 'metal')
cathode = ct.Solution(ctifile, 'metal')
elyte_an_b = ct.Solution(ctifile, 'naf_bulk_an')
elyte_ca_b = ct.Solution(ctifile, 'naf_bulk_ca')
elyte_an_s = ct.Interface(ctifile, 'naf_surf_an', [elyte_an_b, gas_an])
elyte_ca_s = ct.Interface(ctifile, 'naf_surf_ca', [elyte_ca_b, gas_ca])
cat_an_s = ct.Interface(ctifile, 'Pt_surf_an', [anode, elyte_an_b])
cat_ca_s = ct.Interface(ctifile, 'Pt_surf_ca', [cathode, elyte_ca_b])

# Applied current range:
I_app = np.arange(0., 1.0, 0.02)

# Electrolyte resistance [Ohm*m^2]:
R_elyte = 70e-3

# Temperature [K] and pressure [kPa]:
T = 333
P = 1.5*ct.one_atm

# Other constants:
F = ct.faraday

# Areas [m^2]:
SApt_an = 0
SApt_ca = 0

# State state for all phases:
phases = [gas_an, gas_ca, anode, cathode, elyte_an_b, elyte_ca_b, elyte_an_s,
          elyte_ca_s, cat_an_s, cat_ca_s]
for ph in phases:
    ph.TP = T, P
    
# Subfunctions for determining potentials:
def anode_curr(phi_el,I_app,phi_ed,X_elyte):
    # Set the electrolyte composition:
    elyte_an_b.X = X_elyte
    
    # Set the elctrode and electrolyte potentials:
    elyte_an_b.electric_potential = phi_el
    elyte_an_s.electric_potential = phi_el
    anode.electric_potential = phi_ed
    cat_an_s.electric_potential = phi_ed
    
    # Get the net production rate of electrons in the anode (per m^2 Pt):
    r_elec = cat_an_s.net_production_rates[0]
    
    anCurr = r_elec*F*SApt_an
    diff = I_app + anCurr
    
    return diff

def cathode_curr(phi_ed,I_app,phi_el,X_elyte):
    # Set the electrolyte composition:
    elyte_ca_b.X = X_elyte
    
    # Set the electrode and electrolyte potentials:
    elyte_ca_b.electric_potential = phi_el
    elyte_ca_s.electric_potential = phi_el
    cathode.electric_potential = phi_ed
    cat_ca_s.electric_potential = phi_ed
    
    # Get the net production rate of electrons in the cathode (per m^2 Pt):
    r_elec = cat_ca_s.net_production_rates[0]
    
    caCurr = r_elec*F*SApt_ca
    diff = I_app - caCurr
    
    return diff

# Initialize array for potentials:
E_cell_kin = np.zeros_like(I_app)

for i,X_elyte in enumerate(X_elyte):
    # Set anode electrode potential to 0 as reference:
    phi_ed_an = 0
    E_init = 1.0
    
    phi_el_an = fsolve(anode_curr,E_init,args=(I_app,phi_ed_an,X_elyte))
    
    # Calculate electrolyte potential at cathode interface:
    phi_el_ca = phi_el_an + I_app*R_elyte
    
    # Calculate cathode electrode potential:
    phi_ed_ca = fsolve(cathode_curr,E_init,args=(I_app,phi_el_ca,X_elyte))
    
    # Calculate cell voltage:
    E_cell_kin[i] = phi_ed_ca - phi_ed_an