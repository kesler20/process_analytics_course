#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:54:43 2019

@author: Sal Garcia
"""
import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp
import matplotlib.pyplot as plt



# Load the data from Excel
Raman_Spectra     = pd.read_excel('Raman.xlsx', 'Raman', index_col=None, na_values=np.nan)
API_Conc          = pd.read_excel('Raman.xlsx', 'Y', index_col=None, na_values=np.nan)
Tablet_Categories = pd.read_excel('Raman.xlsx', 'Categorical', index_col=None, na_values=np.nan)

#Use pyphi_plots to plot spectra
pp.plot_spectra(Raman_Spectra,plot_title='Raman Spectra',tab_title='Spectra Raw',
             xaxis_label='channel',yaxis_label='a.u')    

# Create a new Pandas Data Frame with SNV(Raman_Spectra)
Raman_Spectra_snv=phi.snv(Raman_Spectra)

#Use pyphi_plots to plot spectra
pp.plot_spectra(Raman_Spectra_snv,plot_title='Raman with SNV',tab_title='Spectra SNV',
             xaxis_label='channel',yaxis_label='a.u')

# Create a new Pandas Data Frame with Pre-processed spectra
# SavGol transform with: 
#   window_size      = 3
#   derivative order = 1
#   polynomial order = 2
Raman_Spectra_snv_savgol,M=phi.savgol(3,1,2,Raman_Spectra_snv)

#Use pyphi_plots to plot spectra
pp.plot_spectra(Raman_Spectra_snv_savgol,plot_title='Raman with SNV and Savitzky Golay',tab_title='Spectra SNV+SavGol',
             xaxis_label='channel',yaxis_label='a.u')

#Build three models and contrast
pls_raman_calibration=phi.pls(Raman_Spectra,API_Conc,3,mcsX='center',mcsY='center')
pp.predvsobs(pls_raman_calibration,Raman_Spectra,API_Conc)

pls_raman_calibration_snv=phi.pls(Raman_Spectra_snv,API_Conc,3,mcsX='center',mcsY='center',cross_val=10)
pp.predvsobs(pls_raman_calibration_snv,Raman_Spectra_snv,API_Conc)

pls_raman_calibration_snv_savgol=phi.pls(Raman_Spectra_snv_savgol,API_Conc,1,mcsX='center',mcsY='center',cross_val=10)
pp.predvsobs(pls_raman_calibration_snv_savgol,Raman_Spectra_snv_savgol,API_Conc,CLASSID=Tablet_Categories,colorby='Type')
pp.predvsobs(pls_raman_calibration_snv_savgol,Raman_Spectra_snv_savgol,API_Conc,CLASSID=Tablet_Categories,colorby='Scale')

preds = phi.pls_pred(Raman_Spectra_snv_savgol,pls_raman_calibration_snv_savgol)
Yhat = preds['Yhat']

plt.figure()
plt.plot(API_Conc['active (%w/w)'].values,Yhat,'o')
RMSE = np.sqrt(np.mean(((API_Conc['active (%w/w)'].values-Yhat)**2)))
plt.title('RMSE: '+str(round(RMSE,3) ))
plt.xlabel('Observed')
plt.ylabel('Predicted')