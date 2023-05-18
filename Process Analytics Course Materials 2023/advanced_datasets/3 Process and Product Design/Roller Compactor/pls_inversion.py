# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 21:02:06 2023

@author: salva
"""

import pyphi as phi
import pyphi_plots as pp
import pandas as pd
import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt
# import os
# #                           feel free to insert your own email
# os.environ['NEOS_EMAIL'] = 'pyphisoftware@gmail.com' 


x_space=pd.read_excel('RCDoEData.xlsx',sheet_name='Process')
y_space=pd.read_excel('RCDoEData.xlsx',sheet_name='SFCS')

plsobj=phi.pls(x_space,y_space,2,cross_val=5)
pp.r2pv(plsobj)
pp.score_scatter(plsobj,[1,2])
pp.loadings_map(plsobj,[1,2])

#%% Invert and show what happens with constraints and illustrate effect of
# null spaces

pls_obj = phi.adapt_pls_4_pyomo(plsobj)
see_solver_diagnostics = True

## Enter equality constraints
#                        SF    CS
yeq         = np.array([0.62, 3.1])
#yeq         = np.array([0.68536497, 3.20596818])
yeq_weights = np.array([1 , 1])
##

# analytical solution
ydes=(yeq-plsobj['my'])/plsobj['sy']
tau_a = np.linalg.inv(plsobj['Q'].T@plsobj['Q'])@plsobj['Q'].T@ydes.reshape(-1,1)
yobt_a=((plsobj['Q']@tau_a)*plsobj['sy'])+plsobj['my']

#Set up PYOMO model
yeq         = phi.np1D2pyomo(yeq)                 #  convert numpy to dictionary
yeq_weights = phi.np1D2pyomo(yeq_weights)         #  convert numpy to dictionary

# initialise the model
model             = ConcreteModel()
# initialise a set for xs and the ys as well as the other components
model.A           = Set(initialize = pls_obj['pyo_A'] )
model.N           = Set(initialize = pls_obj['pyo_N'] )
model.M           = Set(initialize = pls_obj['pyo_M'] )
# add all the variables
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
model.yeq         = Param(model.M, initialize = yeq)
model.yeq_weights = Param(model.M, initialize = yeq_weights)
model.spe_lim    = Param(initialize  = pls_obj['speX_lim95'])
model.ht2lim     = Param(initialize  = pls_obj['T2_lim99'])

def calc_scores(model,i):
    return model.tau[i] == sum(model.Ws[n,i] * ((model.x[n]-model.mx[n])/model.sx[n]) for n in model.N )
model.eq1 = Constraint(model.A,rule=calc_scores)

def y_hat_calc(model,i):
    return (model.y_hat[i]-model.my[i])/model.sy[i]==sum(model.Q[i,a]*model.tau[a] for a in model.A)
model.eq2 = Constraint(model.M,rule=y_hat_calc)

def calc_ht2(model):
    return model.ht2 == sum( model.tau[a]**2/model.var_t[a] for a in model.A)
model.eq3 = Constraint(rule=calc_ht2)

def x_hat_calc(model,i):
    return (model.x_hat[i]-model.mx[i])/model.sx[i]==sum(model.P[i,a]*model.tau[a] for a in model.A)
model.eq4 = Constraint(model.N,rule=x_hat_calc)

# this is the soft constrain which will ask to minimise the hot t^2
def obj_rule(model):
    return sum(model.yeq_weights[m]*(model.yeq[m]-model.y_hat[m])**2 for m in model.M)+0.000*model.ht2
model.obj = Objective(rule=obj_rule) 

# this is a hard constrain which is given a specific value to the optimiser to follow for the hot t^2
# def ht2_lessthan(model):
#     return model.ht2 <= model.ht2lim
# model.ht2_hc_eq = Constraint(rule=ht2_lessthan )


#Solve

#if solving with ipopt locally with MA57
# solver = SolverFactory('ipopt')
# solver.options['linear_solver']='ma57'
# results=solver.solve(model,tee=see_solver_diagnostics)

# If solving with GAMS locally (you need a GAMS license)
solver = SolverFactory('gams')
results=solver.solve(model,tee=see_solver_diagnostics)

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
x_var_names=list(x_space)
x_var_names=x_var_names[1:]
c=0
for x_val in x:
    print(x_var_names[c]+" :"+str(x_val))
    c=c+1
    
    
y_var_names=list(y_space)
y_var_names=y_var_names[1:]
c=0
for y_val in y_hat:
    print(y_var_names[c]+" :"+str(y_val))
    c=c+1

print(tau)

plt.figure()
plt.plot(plsobj['T'][:,0],plsobj['T'][:,1],'oc')
plt.plot(tau[0],tau[1],'ok')
plt.axhline(y=0,color='k',linestyle='--')
plt.axvline(x=0,color='k',linestyle='--')

#Add null space lines
q11=plsobj['Q'][0,0]  #var = 1 A = 1
q12=plsobj['Q'][0,1]  #var = 1 A = 2
q21=plsobj['Q'][1,0]  #var = 2 A = 1
q22=plsobj['Q'][1,1]  #var = 2 A = 2

# Null space for var 1
# ns_1_1 *q11 + ns_1_2*q12 = 0
ns_1_1=np.arange(-8,3)
ns_1_2=-q11*ns_1_1/q12
ns_1_1=ns_1_1+tau_a[0]
ns_1_2=ns_1_2+tau_a[1]

# Null space for var 2
# ns_2_1 *q21 + ns_2_2*q22 = 0
ns_2_1=np.arange(-8,3)
ns_2_2=-q21*ns_2_1/q22
ns_2_1=ns_2_1+tau_a[0]
ns_2_2=ns_2_2+tau_a[1]

#Plot scores
plt.plot(tau[0],tau[1],'sb')
plt.plot(tau_a[0],tau_a[1],'^k')
plt.plot(ns_1_1,ns_1_2,':r')
plt.plot(ns_2_1,ns_2_2,'--r')

#Plot Hot T2 99% Confidence interval
var_t=np.var(plsobj['T'],ddof=1,axis=0)
t1_minmax=np.sqrt(plsobj['T2_lim99']*var_t[0] )
t1_dual=np.linspace(-t1_minmax,t1_minmax,100)
t2_dual_p = np.sqrt((plsobj['T2_lim99']-( (t1_dual**2)/var_t[0]) )*var_t[1])
t2_dual_n = -np.sqrt((plsobj['T2_lim99']-( (t1_dual**2)/var_t[0]) )*var_t[1])
plt.plot(t1_dual,t2_dual_p,':c')
plt.plot(t1_dual,t2_dual_n,':c')


plt.show()