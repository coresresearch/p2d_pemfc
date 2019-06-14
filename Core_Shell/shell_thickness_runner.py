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
from read_write_funcs import *
plt.close('all')




""" Define thickness values to examine and volume fraction relation """
"-----------------------------------------------------------------------------"
generic = 'gasLpt_'                # prefix to all folder names created

t_vec = np.array([25, 10, 5])*1e-9     # Nafion thicknesses [m] (high to low)
#i_vec = np.array([1.253, 2.585, 3.167]) # end i_ext for each thickness [A/cm^2] C high Pt
#i_vec = np.array([1.253, 2.574, 2.938]) # end i_ext for each thickness [A/cm^2] G high Pt

#i_vec = np.array([0.423, 1.585, 2.167]) # end i_ext for each thickness [A/cm^2] C low Pt
#i_vec = np.array([0.423, 0.685, 0.926]) # end i_ext for each thickness [A/cm^2] G low Pt

phase = 0                     # change (0:gas, 1:carbon) phase volume as thickness changes
Ni = np.array([20,50,50])     # linear steps for i_ext2 to make smooth polarization curve




""" Run model over thickness range """
"-----------------------------------------------------------------------------"
# Import all of the user inputs:
from particle_shell_pemfc_runner import *
import particle_shell_pemfc_runner as run
folder_name = run.folder_name = generic + str(int(t_vec[0]*1e9)) + 'nm'
post_only = run.post_only = 0
i_OCV = run.i_OCV = 0
polar = run.polar = 1
save = run.save = 1

# Set the first Nafion thickness and calculate geometries:
t_naf = run.t_naf = t_vec[0]
i_ext2 = run.i_ext2 = np.linspace(i_ext2[0],i_vec[0],Ni[0])
exec(open("particle_shell_pemfc_pre.py").read())
exec(open("particle_shell_pemfc_dsvdt.py").read())

if save == 1:
    ModuleWriter(cwd + '/' + folder_name + '/user_inputs.csv', run)

if phase == 0:
    # Force A_pv_pt and A_pv_dl to be the same, but change gas phase:
    area_calcs = 2
elif phase == 1:
    # Force porosity and A_pv_pt to be the same, but change Carbon phase:
    area_calcs = 3
    
# Run additional Nafion thicknesses w/ differing eps_gas and eps_gas:
for i in range(len(t_vec)-1):
    # name new folder to save each solution for post-processing:
    folder_name = run.folder_name = generic + str(int(t_vec[i+1]*1e9)) + 'nm'
    
    # change end current for thinner t_naf
    i_ext2 = run.i_ext2 = np.linspace(i_ext2[0],i_vec[i+1],Ni[i+1]) 
    
    t_naf = run.t_naf = t_vec[i+1]
    exec(open("particle_shell_pemfc_pre.py").read())
    exec(open("particle_shell_pemfc_dsvdt.py").read())
    
    if save == 1:
        ModuleWriter(cwd + '/' + folder_name + '/user_inputs.csv', run)

