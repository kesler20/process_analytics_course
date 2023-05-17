# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 20:53:21 2022

@author: salva
"""

import pandas as pd
import numpy as np
import pyphi_batch as phibatch
import pyphi_plots as pp
import matplotlib.pyplot as plt
import time

bdata=pd.read_excel('Batch Film Coating.xlsx')


#%% Simplistic alignment simply taking the same number of samples per batch    
bdata_aligned=phibatch.simple_align(bdata,250)
phibatch.plot_var_all_batches(bdata_aligned,
                   plot_title='With simple alignment')

#%% Better alignment taking advantage of the phase information
samples_per_phase={'STARTUP':3, 'HEATING':20,'SPRAYING':40, 
                   'DRYING':40,'DISCHARGING':5}

bdata_aligned_phase=phibatch.phase_simple_align(bdata,samples_per_phase)


#%% Take out unusual batches and fit model to normal data use scores and loadings to understand variations
clean_batch_data=bdata_aligned_phase[np.logical_and(bdata_aligned_phase['BATCH NUMBER']!='B1905', 
                                     bdata_aligned_phase['BATCH NUMBER']!='B1805')]

dev_batch_data=bdata_aligned_phase[np.logical_or(bdata_aligned_phase['BATCH NUMBER']=='B1905', 
                                     bdata_aligned_phase['BATCH NUMBER']=='B1805')]

mpca_obj=phibatch.mpca(clean_batch_data,2, phase_samples=samples_per_phase,cross_val=5)
phibatch.loadings_abs_integral(mpca_obj)
phibatch.loadings(mpca_obj,2,which_var=['INLET_AIR_TEMP','EXHAUST_AIR_TEMP','TOTAL_SPRAY_USED'])

