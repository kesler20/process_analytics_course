# -*- coding: utf-8 -*-
"""
Created on Mon May  2 08:04:39 2022

@author: salva
"""

import scipy.io as spio
import numpy as np
import pyphi as phi
import matplotlib.pyplot as plt
import pandas as pd
import pyphi_plots as pp


#LOAD THE DATA FROM A MATLAB FILE AND PREPARE DATAFRAMES
NMRData      = spio.loadmat('NMR_40wines.mat')
y_columns=[]
for l in NMRData['Label'][0]:
   y_columns.append(str(l[0]))
Y = pd.DataFrame(NMRData['Y'],columns=y_columns)
obs_id=[]
for o in np.arange(1,Y.shape[0]+1):
    obs_id.append('Obs'+str(o))
Y.insert(0,'ObsID',obs_id)
spectra=NMRData['X']
ppm=NMRData['ppm'][0]
ppm_str=[]
for i in ppm:
    ppm_str.append(str(round(i,5)))
spectra=pd.DataFrame(spectra,columns=ppm_str)
spectra.insert(0,'ObsID',obs_id)
##########################################

plt.figure()
plt.plot(ppm,NMRData['X'].T)
plt.xlabel('ppm')

#Need to force nipals or calculation is done with SVD and takes a long time
pls_obj=phi.pls(spectra,Y,2,force_nipals=True)
pp.r2pv(pls_obj)

