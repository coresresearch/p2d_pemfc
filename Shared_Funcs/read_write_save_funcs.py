"""
The following functions were created to read/write variable values from/to .csv
files. They were designed to be used with the formatting used in the PEM fuel
cell models in which variables are stored in modules and dictionaries. Use of 
these functions allows simple saving and reading of these variables regardless
of their storage method.

A SaveFiles function was also added to easily create copies of files used to 
run the PEM fuel cell models. This allows the user to go back and check how the
solution was calculated at that time even if the current version of the model 
has been updated to fix bugs or incorporate additional physics.
"""



""" Read and Write w.r.t. Modules """
"-----------------------------------------------------------------------------"
def ModuleWriter(file, module):
    import types, csv
    
    f = open(file, 'w')
    w = csv.writer(f, lineterminator='\n')
    
    for item in dir(module):
        if not item.startswith("__"):
            if type(vars(module)[item]) != types.ModuleType:
                w.writerow([item, vars(module)[item]])

    f.close()
    
def ModuleReader(file):
    import csv, numpy
    
    f = open(file, 'r')
    reader = csv.reader(f)
     
    d = {}
    for row in reader:
        k, v = row
        
        if '[' not in v:
            try:
                d[k] = eval(v)
            except:
                d[k] = v
        else:
            d[k] = " ".join(v.split()).replace(' ',', ')
            d[k] = numpy.asarray(eval(d[k]))
    
    f.close()
    
    return d
    
    
    
""" Read and Write w.r.t. Dictionaries """
"-----------------------------------------------------------------------------"
def DictWriter(file, dictionary):
    import csv
    
    f = open(file, 'w')
    w = csv.writer(f, lineterminator='\n')
    
    for k,v in dictionary.items():
        w.writerow([k,v])
        
    f.close()
    
def DictReader(file):
    import csv, numpy
    
    f = open(file, 'r')
    reader = csv.reader(f)
     
    p = {}
    for row in reader:
        k, v = row
        
        if '[' not in v:
            p[k] = eval(v)
        else:
            p[k] = " ".join(v.split()).replace(' ',', ')
            p[k] = numpy.asarray(eval(p[k]))
    
    f.close()
    
    return p



""" Save File Copies """
"-----------------------------------------------------------------------------"
def SaveFiles(folder_name, ctifile, p, sv_save, user_inputs):
    import os, sys
    import numpy as np
    import cantera as ct
    from shutil import copy2, rmtree
    
    """ Set up saving location """
    "-------------------------------------------------------------------------"
    cwd = os.getcwd()
    
    # Create folder for any files/outputs to be saved:
    if os.path.exists(folder_name):
        print('\nWARNING: folder_name already exists. Files will be overwritten.')
        print('\n"Enter" to continue and overwrite or "Ctrl+c" to cancel.')
        print('In a GUI, e.g. Spyder, "Ctrl+d" may be needed to cancel.')
        user_in = input()   
        if user_in == KeyboardInterrupt:
            sys.exit(0)
        else:
            rmtree(folder_name)
            os.makedirs(folder_name)
            copy2(cwd + '/../' + 'pemfc_runner.py', folder_name)
            copy2(cwd + '/../' + 'Shared_Funcs/pemfc_pre.py', folder_name)
            copy2(cwd + '/../' + 'Shared_Funcs/pemfc_dsvdt.py', folder_name)
            copy2(cwd + '/../' + 'Shared_Funcs/pemfc_post.py', folder_name)
            copy2(cwd + '/../' + 'Shared_Funcs/pemfc_transport_funcs.py', folder_name)
            copy2(cwd + '/../' + 'Shared_Funcs/pemfc_property_funcs.py', folder_name)
            ModuleWriter(cwd + '/' + folder_name + '/user_inputs.csv', user_inputs)
            
    else:
        os.makedirs(folder_name)
        copy2(cwd + '/../' + 'pemfc_runner.py', folder_name)
        copy2(cwd + '/../' + 'Shared_Funcs/pemfc_pre.py', folder_name)
        copy2(cwd + '/../' + 'Shared_Funcs/pemfc_dsvdt.py', folder_name)
        copy2(cwd + '/../' + 'Shared_Funcs/pemfc_post.py', folder_name)
        copy2(cwd + '/../' + 'Shared_Funcs/pemfc_transport_funcs.py', folder_name)
        copy2(cwd + '/../' + 'Shared_Funcs/pemfc_property_funcs.py', folder_name)
        ModuleWriter(cwd + '/' + folder_name + '/user_inputs.csv', user_inputs)
    
    # Save the current cti files into new folder:
    cti_path = ct.__path__[0]
    if os.path.exists(cwd + '/../Core_Shell/' + ctifile):
        copy2(cwd + '/../Core_Shell/' + ctifile, folder_name)
    elif os.path.exists(cwd + '/../Flooded_Agg/' + ctifile):
        copy2(cwd + '/../Flooded_Agg/' + ctifile, folder_name)
    else:
        copy2(cti_path + '/data/' + ctifile, folder_name)
        
    # Save the current parameters for post processing:
    DictWriter(cwd + '/' + folder_name + '/params.csv', p)
    
    # Save a copy of the full solution matrix:
    np.savetxt(cwd +'/' +folder_name +'/solution.csv', sv_save, delimiter=',')
    