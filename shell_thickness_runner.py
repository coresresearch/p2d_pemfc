"""
t_naf test...:
    This scrips runs the Particle-shell model over a range of Nafion thickness
    values defined below. As the Nafion shell thickness is decreased, the 
    volume is given to the defined phase. Options give the user the ability
    to give volume to either the gas phase or carbon phase. 
    
    The thickest Nafion shell is run first to determine the starting geometric
    parameters for the model. The model gives the user the ability to change
    the porosity as the Nafion shrinks or keep it constant by allowing the 
    carbon phase to take up the extra volume that was previously used for the
    Nafion shell. Each thickness generates a folder will all saved results. 
    These results can be read in from the saved_data_post.py file to provide
    plots and validation.
"""

""" Import needed modules """
"-----------------------------------------------------------------------------"
import numpy as np
import pylab as plt
plt.close('all')




""" Define thickness values to examine and volume fraction relation """
"-----------------------------------------------------------------------------"
folder_generic = 'gas_'             # prefix to all folder names created

t_vec = np.array([25, 10, 5])*1e-9   # Nafion thicknesses [m] (high to low)
#i_vec = np.array([0.83, 2.00, 3.11]) # end i_ext for each thickness [A/cm^2] C
i_vec = np.array([0.83, 1.27, 2.10]) # end i_ext for each thickness [A/cm^2] G

phase = 0  # change (0:gas, 1:carbon) phase volume as thickness changes
Ni = 50    # linear steps for i_ext2 to make smooth polarization curve




""" Run model over thickness range """
"-----------------------------------------------------------------------------"
# Import all of the user inputs:
from particle_shell_pemfc_runner import *
folder_name = folder_generic + '1'
i_OCV = 0
save = 1

# Set the first Nafion thickness and calculate geometries:
t_naf = t_vec[0]
i_ext2 = np.linspace(i_ext2[0],i_vec[0],Ni)
exec(open("particle_shell_pemfc_pre.py").read())
exec(open("particle_shell_pemfc_dsvdt.py").read())

if phase == 0:
    # Force A_pv_pt and A_pv_dl to be the same, but change gas phase:
    area_calcs = 2
elif phase == 1:
    # Force porosity and A_pv_pt to be the same, but change Carbon phase:
    area_calcs = 3
    
# Run additional Nafion thicknesses w/ differing eps_gas and eps_gas:
for i in range(len(t_vec)-1):
    # name new folder to save each solution for post-processing:
    folder_name = folder_generic + str(i+2)
    
    # change end current for thinner t_naf
    i_ext2 = np.linspace(i_ext2[0],i_vec[i+1],Ni) 
    
    t_naf = t_vec[i+1]
    exec(open("particle_shell_pemfc_pre.py").read())
    exec(open("particle_shell_pemfc_dsvdt.py").read())

