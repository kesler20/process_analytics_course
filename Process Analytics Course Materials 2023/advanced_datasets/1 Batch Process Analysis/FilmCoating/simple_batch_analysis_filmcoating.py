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
#plot all variables
phibatch.plot_var_all_batches(bdata)
#plot a variable
phibatch.plot_var_all_batches(bdata,which_var='INLET_AIR_TEMP')
#plot some variables
phibatch.plot_var_all_batches(bdata,which_var=['INLET_AIR_TEMP','EXHAUST_AIR_TEMP'])


#%% Simplistic alignment simply taking the same number of samples per batch    
bdata_aligned=phibatch.simple_align(bdata,250)
phibatch.plot_var_all_batches(bdata_aligned,
                   plot_title='With simple alignment')

#%% Better alignment taking advantage of the phase information
samples_per_phase={'STARTUP':3, 'HEATING':20,'SPRAYING':40, 
                   'DRYING':40,'DISCHARGING':5}

bdata_aligned_phase=phibatch.phase_simple_align(bdata,samples_per_phase)
phibatch.plot_var_all_batches(bdata_aligned_phase,
                   plot_title='Batch data synchronized by phase',
                   phase_samples=samples_per_phase)

#%% Build a model with all batches and use scores to understand spread
mpca_obj=phibatch.mpca(bdata_aligned_phase,2, phase_samples=samples_per_phase)
pp.score_scatter(mpca_obj,[1,2],add_labels=True,marker_size=10)
pp.diagnostics(mpca_obj)
phibatch.r2pv(mpca_obj,which_var='TOTAL_SPRAY_USED')

phibatch.contributions(mpca_obj, bdata_aligned_phase, 'scores',to_obs=['B1905'],dyn_conts=True)
phibatch.contributions(mpca_obj, bdata_aligned_phase, 'scores',to_obs=['B1805'])

phibatch.plot_batch(bdata_aligned_phase,
           which_batch=['B1905'],
           which_var=['INLET_AIR_TEMP','EXHAUST_AIR_TEMP','INLET_AIR'],
           include_mean_exc=True,
           include_set=True,
           phase_samples=samples_per_phase)

phibatch.plot_batch(bdata_aligned_phase,
           which_batch=['B1905','B1805'],
           which_var=['INLET_AIR_TEMP','TOTAL_SPRAY_USED'],
           include_mean_exc=True,
           include_set=True,
           phase_samples=samples_per_phase)
#%% Take out unusual batches and fit model to normal data use scores and loadings to understand variations
clean_batch_data=bdata_aligned_phase[np.logical_and(bdata_aligned_phase['BATCH NUMBER']!='B1905', 
                                     bdata_aligned_phase['BATCH NUMBER']!='B1805')]

dev_batch_data=bdata_aligned_phase[np.logical_or(bdata_aligned_phase['BATCH NUMBER']=='B1905', 
                                     bdata_aligned_phase['BATCH NUMBER']=='B1805')]


mpca_obj=phibatch.mpca(clean_batch_data,2, phase_samples=samples_per_phase,cross_val=5)
pp.score_scatter(mpca_obj,[1,2])
pp.diagnostics(mpca_obj)
phibatch.r2pv(mpca_obj)
phibatch.loadings_abs_integral(mpca_obj)
phibatch.loadings(mpca_obj,2,which_var=['INLET_AIR_TEMP','EXHAUST_AIR_TEMP','TOTAL_SPRAY_USED'])

phibatch.plot_batch(bdata_aligned_phase,
           which_batch=['B1910','B2110'],
           which_var=['TOTAL_SPRAY_USED'],
           include_mean_exc=True,
           include_set=True,
           phase_samples=samples_per_phase,single_plot=True)

batch_list=['B1910','B1205','B2510','B2210','B1810','B2110']
phibatch.plot_batch(bdata_aligned_phase,
           which_batch=batch_list,
           which_var=['EXHAUST_AIR_TEMP'],
           include_mean_exc=True,
           include_set=True,
           phase_samples=samples_per_phase,single_plot=True)

phibatch.plot_batch(bdata_aligned_phase,
           which_batch=batch_list,
           which_var=['INLET_AIR_TEMP'],
           include_mean_exc=True,
           include_set=True,
           phase_samples=samples_per_phase,single_plot=True)

#%% Prepare the model for monitorig

phibatch.monitor(mpca_obj, clean_batch_data)
#%%Monitor  all normal batches and show diagnostics
all_batches=np.unique(clean_batch_data['BATCH NUMBER'].values.tolist())
mon_all_batches=phibatch.monitor(mpca_obj,clean_batch_data,
                                 which_batch=all_batches)

#%% Monitor batch 1905 and diagnose

mon_1905 = phibatch.monitor(mpca_obj,dev_batch_data,which_batch=['B1905'])

#contribution to instantaneous SPE at sample 5
sam_num = 4
plt.figure()
plt.bar(mon_1905['cont_spei'].columns ,mon_1905['cont_spei'].iloc[sam_num])
plt.xticks(rotation=90)
plt.ylabel('Contributions to i-SPE')
plt.title('Contributions to instantaneous up to sample #'+str(sam_num))
plt.tight_layout()


phibatch.plot_batch(bdata_aligned_phase,
           which_batch=['B1905'],
           which_var=['INLET_AIR_TEMP','EXHAUST_AIR_TEMP','INLET_AIR'],
           include_mean_exc=True,
           include_set=True,
           phase_samples=samples_per_phase)
#%% Prepare forecasting for batch B2510
batch2forecast='B2510'
mon_data=phibatch.monitor(mpca_obj,clean_batch_data,which_batch=batch2forecast)
#%% Plot forecast for the inlet_temp
var='INLET_AIR_TEMP'
point_in_time=40
forecast=mon_data['forecast']
mdata=clean_batch_data[clean_batch_data['BATCH NUMBER']==batch2forecast]

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

    

#%%small example on variable wise

mpca_obj_vw=phibatch.mpca(bdata_aligned_phase,2,unfolding='variable wise', phase_samples=samples_per_phase)
pp.loadings(mpca_obj_vw)
scores_df=pd.DataFrame(mpca_obj_vw['T'],columns=['PC1','PC2'],index=mpca_obj_vw['obsidX'])
plt.figure()
for b in np.unique(scores_df.index.tolist()):
    plt.plot(scores_df['PC1'][scores_df.index==b].values)
plt.xlabel('samples')
plt.ylabel('PC[1]')
plt.title('Scores per batch for variable wise unfolding')

plt.figure()
for b in np.unique(scores_df.index.tolist()):
    plt.plot(scores_df['PC2'][scores_df.index==b].values)
plt.xlabel('samples')
plt.ylabel('PC[2]')
plt.title('Scores per batch for variable wise unfolding')





    