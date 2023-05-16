
"""
Created on Sat Apr  1 19:54:20 2023

@author: sal
"""
import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp
import matplotlib.pyplot as plt
import pyomo.environ as pyomo
import os

# Feel free to use your email
os.environ['NEOS_EMAIL'] = 'pyphisoftware@gmail.com' 


data=pd.read_excel('BioPolymerData.xlsx',sheet_name='USETHISONE')

xcols=['Experiment', 'NIPAM [mol fraction]',
       'AA [mol fraction]', 'VC [mol fraction]', 'AIBMe [mol fraction]',
       'TGA [mol fraction]', 'AESH [mol fraction]', 'EtOH BP 78C [mL/mol]',
       'DMF BP 153 C [mL/mol]', 'THF BP 66 C [mL]',
       'Reaction temperature [°C]', 'Reaction time [h]',
       'Dialysis membrane MWCO [Da]']
ycols=['Experiment','Cloud point [°C]','Avg. No. Mol. Wt., Mn [Da]','Recovery [%]']

X=data[xcols]
Y=data[ycols]

pls_obj=phi.pls(X,Y,6,cross_val=5,cross_val_X=True)
pp.r2pv(pls_obj,addtitle='PLS')




