"""
Description:
    This scrips runs the PEMFC models over a range of Nafion shell thickness
    values defined below. As the Nafion shell thickness is decreased, the 
    volume is given to the defined phase. Options give the user the ability
    to give volume to either the gas phase or carbon phase. 
    
    The thickest Nafion shell is run first to determine the starting geometric
    parameters for the model. The model gives the user the ability to change
    the porosity as the Nafion shrinks or keep it constant by allowing the 
    carbon phase to take up the extra volume that was previously used for the
    Nafion shell. Each thickness generates a folder with all saved results. 
    These results can be read in from the saved_data_post.py file to provide
    plots and validation.
    
    Since the property models implemented here show that sig_io and D_O2 change
    as a function of t_naf, the PEMFC performance is also affected. Specifically,
    both of these properties increase with increasing t_naf which suggests 
    thicker shells may have better performance under certain conditions.
    
Notes:
    Currently, the geometry calculations are only set up for this looping t_naf
    runner to be used with the core-shell model. If using the flooded-agg model,
    the results will only be accurate for the first shell thickness, since the 
    calculations required to change the geometry have not yet been implemented
    for this model.
    
    This script is meant to simply loop through different Nafion shell thickness
    values. All of the other geometry input values should be set in the file
    pemfc_runner.py.
"""

""" Import needed modules """
"-----------------------------------------------------------------------------"
import os
import numpy as np
import pylab as plt

os.chdir(os.getcwd() + '/../')
from Shared_Funcs.read_write_save_funcs import *

plt.close('all')

""" Define thickness values to examine and volume fraction relation """
"-----------------------------------------------------------------------------"
generic = 'gasLpt_'           # prefix to all folder names created

t_vec = np.array([25, 10, 5])*1e-9      # Nafion thicknesses [m] (high to low)
i_vec = np.array([1.2, 1.2, 1.2])
#i_vec = np.array([1.253, 2.585, 3.167]) # end i_ext for each thickness [A/cm^2] C high Pt
#i_vec = np.array([1.253, 2.574, 2.938]) # end i_ext for each thickness [A/cm^2] G high Pt

#i_vec = np.array([0.423, 1.585, 2.167]) # end i_ext for each thickness [A/cm^2] C low Pt
#i_vec = np.array([0.423, 0.685, 0.926]) # end i_ext for each thickness [A/cm^2] G low Pt

phase = 0                     # change (0:gas, 1:carbon) phase volume as thickness changes
Ni = np.array([5,5,5])        # linear steps for i_ext2 to make smooth polarization curve

""" Run model over thickness range """
"-----------------------------------------------------------------------------"
# Import all of the user inputs:
from pemfc_runner import *
import pemfc_runner as user_inputs
folder_name = user_inputs.folder_name = generic + str(int(t_vec[0]*1e9)) + 'nm'
post_only = user_inputs.post_only = 0
i_OCV = user_inputs.i_OCV = 0
save = user_inputs.save = 1

# Set the first Nafion thickness and calculate geometries:
t_naf = user_inputs.t_naf = t_vec[0]
i_ext2 = user_inputs.i_ext2 = np.linspace(i_ext2[0],i_vec[0],Ni[0])
exec(open("Shared_Funcs/pemfc_pre.py").read())
exec(open("Shared_Funcs/pemfc_dsvdt.py").read())
exec(open("Shared_Funcs/pemfc_post.py").read())
if save == 1:
    ModuleWriter(cwd +'/Saved_Results/' +folder_name +'/user_inputs.csv', user_inputs)

if phase == 0:
    # Force A_pv_pt and A_pv_dl to be the same, but change gas phase:
    area_calcs = 2
elif phase == 1:
    # Force porosity and A_pv_pt to be the same, but change Carbon phase:
    area_calcs = 3
    
# Run additional Nafion thicknesses w/ differing eps_gas and eps_gas:
for i in range(len(t_vec)-1):
    # name new folder to save each solution for post-processing:
    folder_name = user_inputs.folder_name = generic + str(int(t_vec[i+1]*1e9)) + 'nm'
    
    # change end current for thinner t_naf
    i_ext2 = user_inputs.i_ext2 = np.linspace(i_ext2[0],i_vec[i+1],Ni[i+1]) 
    
    t_naf = user_inputs.t_naf = t_vec[i+1]
    exec(open("Shared_Funcs/pemfc_pre.py").read())
    exec(open("Shared_Funcs/pemfc_dsvdt.py").read())
    exec(open("Shared_Funcs/pemfc_post.py").read())
    if save == 1:
        ModuleWriter(cwd +'/Saved_Results/' +folder_name +'/user_inputs.csv', user_inputs)