# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 22:33:38 2022

@author: salva
"""


import pandas as pd
import numpy as np
import pyphi_batch as phibatch
import matplotlib.pyplot as plt
import pyphi_plots as pp
        

bdata        = pd.read_excel('Batch Dryer Case Study.xlsx',sheet_name='Trajectories')
cqa          = pd.read_excel('Batch Dryer Case Study.xlsx',sheet_name='ProductQuality')
char         = pd.read_excel('Batch Dryer Case Study.xlsx',sheet_name='ProcessCharacteristics')
initial_chem = pd.read_excel('Batch Dryer Case Study.xlsx',sheet_name='InitialChemistry')
cat          = pd.read_excel('Batch Dryer Case Study.xlsx',sheet_name='classifiers')

phibatch.plot_var_all_batches(bdata)
#%%
samples={'Deagglomerate':20,'Heat':30,'Cooldown':40};
bdata_aligned_phase=phibatch.phase_simple_align(bdata,samples)
phibatch.plot_var_all_batches(bdata_aligned_phase,
                  plot_title='simple alignment /phase',
                  phase_samples=samples)
#%%
mpca_obj=phibatch.mpca(bdata_aligned_phase,3,phase_samples=samples)
pp.score_scatter(mpca_obj,[1,2],CLASSID=cat,colorby='Quality 2')

#%% Show a forecast
mon_data =phibatch.monitor(mpca_obj, bdata_aligned_phase,which_batch='Batch 5')

#Show forecast of x
var='Dryer Temp' 
point_in_time=20
batch2forecast='Batch 5'
forecast=mon_data['forecast']
mdata=bdata_aligned_phase[bdata_aligned_phase['Batch number']==batch2forecast]

plt.figure()
f=forecast[point_in_time]
plt.plot(mdata[var][:point_in_time],'o',label='Measured')
aux=[np.nan]*point_in_time
aux.extend(f[var].values[point_in_time:].tolist())
plt.plot(np.array(aux),label='Forecast')
plt.plot(mdata[var][point_in_time:],'o',label='Known trajectory',alpha=0.3)
plt.xlabel('sample')
plt.ylabel(var)
plt.legend()
plt.title('Forecast for '+var+' at sample '+str(point_in_time)+' for '+batch2forecast )

#%% PLS with trajectories
mpls_obj_t=phibatch.mpls(bdata_aligned_phase,cqa,5,phase_samples=samples)
pp.r2pv(mpls_obj_t)
pp.score_scatter(mpls_obj_t, [1,2],CLASSID=cat,colorby='Quality')

#%% PLS with initial chemistry
mpls_obj_i=phibatch.mpls(bdata_aligned_phase,cqa,5,zinit=initial_chem,
                       phase_samples=samples)
pp.score_scatter(mpls_obj_i, [1,2],CLASSID=cat,colorby='Quality')

#%% PLS with initial chemistry MB

mpls_obj_i_mb=phibatch.mpls(bdata_aligned_phase,cqa,5,zinit=initial_chem,
                      phase_samples=samples,mb_each_var='True')
pp.score_scatter(mpls_obj_i_mb, [1,2],CLASSID=cat,colorby='Quality')
pp.mb_weights(mpls_obj_i_mb)
phibatch.loadings_abs_integral(mpls_obj_i_mb)
phibatch.loadings(mpls_obj_i_mb,1)
#What process reccomendation would we do to minimize off-spec product ?

plt.show()