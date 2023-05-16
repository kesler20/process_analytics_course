
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

#%% Run a design excercise with a plain PLS and a MBPLS and see differences.

cloud_point_max = 37
cloud_point_min = 18
mn_max          = 34000

solvent_densities ={'EtOH BP 78C [mL/mol]':0.7892,'DMF BP 153 C [mL/mol]':0.944, 'THF BP 66 C [mL]':0.889 }
solvent_mw        ={'EtOH BP 78C [mL/mol]':46.07,'DMF BP 153 C [mL/mol]':73.09, 'THF BP 66 C [mL]':72.11 }


#Condition matrices as dictionaries
pls_obj = phi.adapt_pls_4_pyomo(pls_obj,use_var_ids=True)
materials   = ['NIPAM [mol fraction]', 'AA [mol fraction]', 'VC [mol fraction]', 
               'AIBMe [mol fraction]','TGA [mol fraction]', 'AESH [mol fraction]']
solvents    = ['EtOH BP 78C [mL/mol]','DMF BP 153 C [mL/mol]', 'THF BP 66 C [mL]' ]
process_var = ['Reaction temperature [°C]', 'Reaction time [h]','Dialysis membrane MWCO [Da]']
xvars = pls_obj['pyo_N']
yvars = pls_obj['pyo_M']
model             = pyomo.ConcreteModel()
model.A           = pyomo.Set(initialize = pls_obj['pyo_A'] )  # Index for LV's
model.mat         = pyomo.Set(initialize = materials)          # Index for Materials
model.sol         = pyomo.Set(initialize = solvents)           # Index for Solvents
model.M           = pyomo.Set(initialize =pls_obj['pyo_M'])    # Index for Y space
model.N           = pyomo.Set(initialize =pls_obj['pyo_N'])    # Index for X space

model.y_hat       = pyomo.Var  (model.M, within=pyomo.NonNegativeReals)
model.Q           = pyomo.Param(model.M,model.A,initialize = pls_obj['pyo_Q'])
model.my          = pyomo.Param(model.M,initialize = pls_obj['pyo_my'])
model.sy          = pyomo.Param(model.M,initialize = pls_obj['pyo_sy'])

model.x_hat       = pyomo.Var(model.N, within=pyomo.Reals)
model.x           = pyomo.Var(model.N, within=pyomo.NonNegativeReals)

model.tau         = pyomo.Var(model.A,within =pyomo.Reals)
model.spe_x       = pyomo.Var(within = pyomo.Reals)
model.ht2         = pyomo.Var(within = pyomo.Reals)
model.var_t       = pyomo.Param(model.A,initialize = pls_obj['pyo_var_t'])
model.spe_lim     = pyomo.Param(initialize  = pls_obj['speX_lim95'])
model.ht2_lim     = pyomo.Param(initialize  = pls_obj['T2_lim95'])

model.Ws          = pyomo.Param(model.N,model.A,initialize = pls_obj['pyo_Ws'])
model.P           = pyomo.Param(model.N,model.A,initialize = pls_obj['pyo_P'])
model.mx          = pyomo.Param(model.N,initialize = pls_obj['pyo_mx'])
model.sx          = pyomo.Param(model.N,initialize = pls_obj['pyo_sx'])

model.molXsol     = pyomo.Var(model.sol,within=pyomo.NonNegativeReals)  #Mol fraction per solvent 
model.molsol      = pyomo.Var(model.sol,within=pyomo.NonNegativeReals)  #Mols of solvent per mol of solutes
model.bpsolv      = pyomo.Var(within=pyomo.NonNegativeReals)            #Boiling point of solvent mixture
model.dens        = pyomo.Param(model.sol, initialize=solvent_densities) #solvent densities
model.mws         = pyomo.Param(model.sol, initialize=solvent_mw)        #solvent molecular weights




