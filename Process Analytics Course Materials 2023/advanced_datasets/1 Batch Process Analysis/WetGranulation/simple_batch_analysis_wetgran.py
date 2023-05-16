# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 20:53:21 2022

@author: salva
"""

import pandas as pd
import numpy as np
import pyphi_batch as phibatch
import matplotlib.pyplot as plt
import pyphi_plots as pp
        

bdata = pd.read_excel('WetGranBatchData.xlsx')
cqa   = pd.read_excel('WetGranBatchData.xlsx',sheet_name='CQAs')
#phibatch.plot_var_all_batches(bdata)
#%%    

bdata_aligned=phibatch.simple_align(bdata,100)
phibatch.plot_var_all_batches(bdata_aligned,
                   plot_title='simple alignment')
#%%
samples={'DRY MIXING':5,'SPRAYING':50,'WET MASSING':10};
bdata_aligned_phase=phibatch.phase_simple_align(bdata,samples)
phibatch.plot_var_all_batches(bdata_aligned_phase,
                   plot_title='simple alignment /phase',
                   phase_samples=samples)
mpls_obj=phibatch.mpls(bdata_aligned_phase,cqa,5,phase_samples=samples)

phibatch.loadings(mpls_obj,1)
pp.r2pv(mpls_obj)

#mpls_obj=phibatch.mpls(bdata_aligned_phase,cqa,5,phase_samples=samples,cross_val=5)

#mpls_obj=phibatch.mpls(bdata_aligned_phase,cqa,5,phase_samples=samples,cross_val=5,cross_val_X=True)
    
phibatch.monitor(mpls_obj, bdata_aligned_phase)   
mon_Batch01=phibatch.monitor(mpls_obj, bdata_aligned_phase,which_batch='Batch01')   
    