    
"""
Data taken from
    
    Dyrby, M., Engelsen, S.B., Nørgaard, L., Bruhn, M. and Lundsberg-Nielsen, L., 2002. 
    Chemometric quantitation of the active substance (containing C≡ N) in a pharmaceutical 
    tablet using near-infrared (NIR) transmittance and NIR FT-Raman spectra. 
    Applied Spectroscopy, 56(5), pp.579-585.
    
    Raw data available from: http://www.models.life.ku.dk/Tablets
    
    
"""

import scipy.io as spio
import numpy as np
import pyphi as phi
import matplotlib.pyplot as plt


#LOAD THE DATA FROM A MATLAB FILE
NIRData      = spio.loadmat('NIR_Dyrby_et_al.mat')
nir_spectra  = np.array(NIRData['spectra'])
Y           = np.array(NIRData['Ck'][:,1])
dose_source  = np.array(NIRData['dose_source'])
wavenumbers  = np.array(NIRData['wavenumber'])
wavenumbers=wavenumbers.reshape(-1)

# PRE-PROCESS SPECTRA 
nir_spectra = phi.snv(nir_spectra)

ws=3
op=2
od=1
nir_spectra,M = phi.savgol(ws,od,op,nir_spectra)
plt.figure()
plt.plot(wavenumbers[ws:-ws], nir_spectra.T)
plt.xlabel('wavenumber')
plt.ylabel('AU')


# Divide the set into Calibration and Validation taking one in every two samples
spectra_cal = nir_spectra[::2,:]
spectra_val = nir_spectra[1:nir_spectra.shape[0]:2,:]
Y_cal       = Y[::2].reshape(-1,1)
Y_val       = Y[1:len(Y):2].reshape(-1,1)
dose_source_cal       = dose_source[::2,:]
dose_source_val       = dose_source[1:dose_source.shape[0]:2,:]

max_components = 5
RMSE=[]
for a in np.arange(max_components)+1:
    pls_obj  = phi.pls(spectra_cal,Y_cal,a,mcsX='center')
    val_pred = phi.pls_pred(spectra_val,pls_obj)
    val_pred = val_pred['Yhat']
    error    = val_pred - Y_val
    RMSE.append ( np.sqrt(np.mean(error**2)))

plt.figure()
plt.plot(np.arange(1,max_components+1),RMSE,'o')    
plt.xlabel('Number of LVs')
plt.ylabel('RMSE')
plt.tight_layout()