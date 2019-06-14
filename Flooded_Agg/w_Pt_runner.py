# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:33:25 2019

@author: Corey Randall
"""

w_Pt_vec = np.array([0.2, 0.1, 0.05, 0.025])
for i, w in enumerate(w_Pt_vec):
    w_Pt = w
    exec(open("flooded_agg_pemfc_runner.py").read())