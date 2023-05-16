# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 19:13:31 2023

@author: salva
"""
import numpy as np
from pyomo.environ import *
import pandas as pd
import pyphi as phi
import pyphi_plots as pp
import os
import matplotlib.pyplot as plt
#                           feel free to insert your own email
os.environ['NEOS_EMAIL'] = 'pyphisoftware@gmail.com' 

sdd_process=pd.read_excel('Spray Drying.xlsx',sheet_name='DryerConditions')
sdd_material=pd.read_excel('Spray Drying.xlsx',sheet_name='SDDMaterial')
tags = pd.read_excel('Spray Drying.xlsx',sheet_name='tags')
plsobj = phi.pls(sdd_process,sdd_material,3,cross_val=20,cross_val_X=True)
pp.r2pv(plsobj)
pp.score_scatter(plsobj,[1,2],CLASSID=tags,colorby='nozzle_diameter')

#%%
target_buk_dens = 0.15
nozzle_diam     = 1.06

pls_obj = phi.adapt_pls_4_pyomo(plsobj,use_var_ids=True)
see_solver_diagnostics = True

## Enter equality constraints


model             = ConcreteModel()
model.A           = Set(initialize = pls_obj['pyo_A'] )
model.N           = Set(initialize = pls_obj['pyo_N'] )
model.M           = Set(initialize = pls_obj['pyo_M'] )
model.y_hat       = Var(model.M, within=Reals)
model.x_hat       = Var(model.N, within=Reals)
model.x           = Var(model.N, within=Reals)
model.tau         = Var(model.A,within = Reals)
model.spe_x       = Var(within = Reals)
model.ht2         = Var(within = Reals)
model.Ws          = Param(model.N,model.A,initialize = pls_obj['pyo_Ws'])
model.P           = Param(model.N,model.A,initialize = pls_obj['pyo_P'])
model.Q           = Param(model.M,model.A,initialize = pls_obj['pyo_Q'])
model.mx          = Param(model.N,initialize = pls_obj['pyo_mx'])
model.sx          = Param(model.N,initialize = pls_obj['pyo_sx'])
model.my          = Param(model.M,initialize = pls_obj['pyo_my'])
model.sy          = Param(model.M,initialize = pls_obj['pyo_sy'])
model.var_t       = Param(model.A,initialize = pls_obj['pyo_var_t'])

model.spe_lim    = Param(initialize  = pls_obj['speX_lim95'])

    
def calc_scores_(model,i):
    return model.tau[i] == sum(model.Ws[n,i] * ((model.x[n]-model.mx[n])/model.sx[n]) for n in model.N )
model.scoreseq = Constraint(model.A,rule=calc_scores_)

def y_hat_calc(model,i):
    return (model.y_hat[i]-model.my[i])/model.sy[i]==sum(model.Q[i,a]*model.tau[a] for a in model.A)
model.yhateq = Constraint(model.M,rule=y_hat_calc)

def calc_ht2(model):
    return model.ht2 == sum( model.tau[a]**2/model.var_t[a] for a in model.A)
model.ht2eq = Constraint(rule=calc_ht2)

def x_hat_calc(model,i):
    return (model.x_hat[i]-model.mx[i])/model.sx[i]==sum(model.P[i,a]*model.tau[a] for a in model.A)
model.xhateq = Constraint(model.N,rule=x_hat_calc)

def calc_spe_x(model):
    return model.spe_x == sum((((model.x[i]-model.mx[i])/model.sx[i] - (model.x_hat[i]-model.mx[i])/model.sx[i])**2) for i in model.N )
model.spexeq = Constraint(rule=calc_spe_x)

def spe_lessthan(model):
    return model.spe_x <= model.spe_lim
model.spelt = Constraint(rule=spe_lessthan)

def nozzle_(model):
     return model.x['nozzle_diameter']==1.06
model.con1 = Constraint(rule=nozzle_)

def obj_rule(model):
    return (1*sum((target_buk_dens-model.y_hat['bulk_dens'])**2 for m in model.M) + 1E-6*model.ht2 )
model.obj = Objective(rule=obj_rule) 
    
#Solve

#if solving with ipopt locally with MA57
solver = SolverFactory('ipopt')
solver.options['linear_solver']='ma57'
results=solver.solve(model,tee=see_solver_diagnostics)

#If solving with GAMS locally (you need a GAMS license)
#solver = SolverFactory('gams:ipopt')
#results=solver.solve(model,tee=see_solver_diagnostics)

#Use these lines to solve with NEOS freebie ! 
#solver_manager = SolverManagerFactory('neos')
#results = solver_manager.solve(model, opt='ipopt', tee=True)

    
x_hat = []
for i in model.x_hat:
    x_hat.append(value(model.x_hat[i]))  
    
x = []
for i in model.x:
    x.append(value(model.x[i]))  
y_hat = []
for i in model.y_hat:
    y_hat.append(value(model.y_hat[i]))  
y_hat   = np.array(y_hat)
x_hat   = np.array(x_hat)
tau = []
for i in model.tau:
    tau.append(value(model.tau[i]))       
tau   = np.array(tau)
tau   = tau.reshape(-1,1)
x_var_names=list(sdd_process)
x_var_names=x_var_names[1:]
print('Process Conditions')
c=0
for x_val in x:
    print(x_var_names[c]+" :"+str(x_val))
    c=c+1
print('\n')    
print('Product Properties')    
y_var_names=list(sdd_material)
y_var_names=y_var_names[1:]
c=0
for y_val in y_hat:
    print(y_var_names[c]+" :"+str(y_val))
    c=c+1

plt.figure()
plt.plot( plsobj['T'][:,0], plsobj['T'][:,1],'om',label='model')
plt.plot(tau[0],tau[1],'ok',label='Solution found')
plt.axvline(x=0,color='k')
plt.axhline(y=0,color='k')
plt.xlabel('t[1]')
plt.ylabel('t[2]')
plt.legend()
print('\n')
print('Solution diagnostics')
print('HT2 Obt: '+str( np.round(value(model.ht2),3) )+' vs limit of '+str( np.round(plsobj['T2_lim95'],3)  ))
print('speX Obt: '+str( np.round(value(model.spe_x),3) )+' vs limit of '+str( np.round(plsobj['speX_lim99'],3)  ))
