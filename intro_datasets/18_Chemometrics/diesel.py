# -*- coding: utf-8 -*-
"""
Created on Mon May  2 06:38:34 2022

@author: salva
"""

import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp
import matplotlib.pyplot as plt

diesel_nir_spectra=pd.read_excel('diesel_example.xlsx','diesel_spec')
diesel_props=pd.read_excel('diesel_example.xlsx','diesel_props')
wv = np.array(diesel_nir_spectra.columns[1:].values,dtype=float)

cols=diesel_nir_spectra.columns
new_cols=[cols[0]]
for c in cols[1:]:
    new_cols.append(str(c))
diesel_nir_spectra=diesel_nir_spectra.set_axis(new_cols,axis=1)
plt.figure()
plt.plot(wv,diesel_nir_spectra.values[:,1:].T)

pls_obj=phi.pls(diesel_nir_spectra,diesel_props,3,mcsX='center')
pp.r2pv(pls_obj,addtitle='Raw Spectra')

diesel_nir_spectra_snv=phi.snv(diesel_nir_spectra)
plt.figure()
plt.plot(wv,diesel_nir_spectra_snv.values[:,1:].T)
plt.title('With SNV')
pls_obj_snv=phi.pls(diesel_nir_spectra_snv,diesel_props,3,mcsX='center')
pp.r2pv(pls_obj_snv,addtitle='SNV')

diesel_nir_spectra_savgol,D=phi.savgol(10,1,2,diesel_nir_spectra)
aux=diesel_nir_spectra_savgol.values[:,1:].astype(float)
plt.figure()
plt.plot(aux.T)
plt.title('With SavGol')
pls_obj_snv=phi.pls(diesel_nir_spectra_savgol,diesel_props,3,mcsX='center')
pp.r2pv(pls_obj_snv,addtitle='SavGol_10_1_2')
        