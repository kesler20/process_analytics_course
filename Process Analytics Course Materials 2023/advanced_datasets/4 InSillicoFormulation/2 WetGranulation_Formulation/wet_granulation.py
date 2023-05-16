# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 19:54:20 2023

@author: salva
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

jr,mats=phi.parse_materials('WetGranDataSet_GhentU.xlsx','Materials')
R=jr[0]
X=pd.read_excel( 'WetGranDataSet_GhentU.xlsx',sheet_name='MatProperties')
Y=pd.read_excel('WetGranDataSet_GhentU.xlsx',sheet_name='Product' )
Z=pd.read_excel('WetGranDataSet_GhentU.xlsx',sheet_name='Process' )

lpls_obj=phi.lpls(X,R,Y,3)
#Notice correspondance of experiments wrt location of materials in their own scoreplot
pp.r2pv(lpls_obj,addtitle='LPLS')
pp.score_scatter(lpls_obj,[1,2],add_labels=True,addtitle='Blends')
pp.score_scatter(lpls_obj,[1,2],add_labels=True,rscores=True,addtitle=' Materials')
pp.loadings_map(lpls_obj, [1,2])


#Create RXI
Xvals=X.values[:,1:].astype(float)
Rvals=R.values[:,1:].astype(float)
#Calculate weighted average physprops and build DataFrame
RXi=Rvals @ Xvals

RXi_df=pd.DataFrame(RXi,columns=X.columns[1:])
RXi_df.insert(0,R.columns[0],R[R.columns[0]])
#Build MBPLS and PLS
#%%
XMB={'Process':Z,'Materials':RXi_df}

Xaug=pd.concat([Z,RXi_df[RXi_df.columns[1:]] ],axis=1)
plsobj=phi.pls(Xaug,Y,3)
mbplsobj=phi.mbpls(XMB,Y,3)
pp.r2pv(plsobj,addtitle='PLS')
pp.r2pv(mbplsobj,addtitle='MBPLS')
#show that the MB scaling is included into sx so you treat it like any other PLS from the numeric standpoint

#%% Run a design excercise with a plain PLS and a MBPLS and see differences.
pls_obj=mbplsobj.copy()
hr_target  = 1.17
fri_target = 25

#Condition matrices as dictionaries
pls_obj = phi.adapt_pls_4_pyomo(pls_obj,use_var_ids=True)
materials   = list(R)[1:]
properties  = list(X)[1:]
process_var = list(Z)[1:]
Xpyo       = dict(((materials[i],properties[j]), Xvals[i][j]) for i in range(Xvals.shape[0]) for j in range(Xvals.shape[1]))


model             = pyomo.ConcreteModel()
model.A           = pyomo.Set(initialize = pls_obj['pyo_A'] )  # Index for LV's
model.z           = pyomo.Set(initialize = process_var)        # Index for process conditions  (Z portion of X)
model.p           = pyomo.Set(initialize = properties )        # Index for Material Properties (RXI portion of X)
model.M           = pyomo.Set(initialize = materials)          # Index for Materials

model.zp          = pyomo.Set(initialize = process_var+properties ) #X Columns Index  - Model X parameters follow this one
model.N           = pyomo.Set(initialize = list(Y)[1:] )            #Y Columns Index  - Model Y parameters follow this one

model.y_hat       = pyomo.Var  (model.N, within=pyomo.Reals)
model.Q           = pyomo.Param(model.N,model.A,initialize = pls_obj['pyo_Q'])
model.my          = pyomo.Param(model.N,initialize = pls_obj['pyo_my'])
model.sy          = pyomo.Param(model.N,initialize = pls_obj['pyo_sy'])


model.zrxi_hat    = pyomo.Var(model.zp, within=pyomo.Reals)
model.zrxi        = pyomo.Var(model.zp, within=pyomo.NonNegativeReals)

model.r           = pyomo.Var(model.M, within=pyomo.NonNegativeReals) #blending info
model.rbin        = pyomo.Var(model.M ,within=pyomo.Binary)

model.tau         = pyomo.Var(model.A,within =pyomo.Reals)
model.spe_zrxi    = pyomo.Var(within = pyomo.Reals)
model.ht2         = pyomo.Var(within = pyomo.Reals)
model.var_t       = pyomo.Param(model.A,initialize = pls_obj['pyo_var_t'])
model.spe_lim     = pyomo.Param(initialize  = pls_obj['speX_lim95'])
model.ht2_lim     = pyomo.Param(initialize  = pls_obj['T2_lim95'])

model.Ws          = pyomo.Param(model.zp,model.A,initialize = pls_obj['pyo_Ws'])
model.P           = pyomo.Param(model.zp,model.A,initialize = pls_obj['pyo_P'])
model.mx          = pyomo.Param(model.zp,initialize = pls_obj['pyo_mx'])
model.sx          = pyomo.Param(model.zp,initialize = pls_obj['pyo_sx'])

model.X          = pyomo.Param(model.M,model.p,initialize  = Xpyo)

def binary_constraint_(model,i):
        return model.r[i]<=model.rbin[i]
model.binary_constraint = pyomo.Constraint(model.M,rule=binary_constraint_)

def only_one_api_const_(model):
    return sum(model.rbin[i] for i in model.M)==2
model.only_one_api_const = pyomo.Constraint(rule=only_one_api_const_)

def sum_unity_(model):
    return sum(model.r[i] for i  in model.M)==1
model.sum_unity = pyomo.Constraint(rule=sum_unity_)

def calc_rxi_(model,i):
    return model.zrxi[i] == sum(model.X[m,i]*model.r[m]  for m in model.M)
model.calc_rxi= pyomo.Constraint(model.p,rule=calc_rxi_ )

def calc_scores_(model,i):
    return model.tau[i] == sum(model.Ws[n,i] * ((model.zrxi[n]-model.mx[n])/model.sx[n]) for n in model.zp )
model.calc_scores = pyomo.Constraint(model.A,rule=calc_scores_)

def y_hat_calc_(model,i):
    return (model.y_hat[i]-model.my[i])/model.sy[i]==sum(model.Q[i,a]*model.tau[a] for a in model.A)
model.y_hat_calc = pyomo.Constraint(model.N,rule=y_hat_calc_)

def calc_ht2_(model):
    return model.ht2 == sum( model.tau[a]**2/model.var_t[a] for a in model.A)
model.calc_ht2 = pyomo.Constraint(rule=calc_ht2_)

def zrxi_hat_calc_(model,i):
    return (model.zrxi_hat[i]-model.mx[i])/model.sx[i]==sum(model.P[i,a]*model.tau[a] for a in model.A)
model.zrxi_hat_calc = pyomo.Constraint(model.p,rule=zrxi_hat_calc_)

def calc_spe_zrxi_(model):
    return model.spe_zrxi == sum((((model.zrxi[i]-model.mx[i])/model.sx[i] - (model.zrxi_hat[i]-model.mx[i])/model.sx[i])**2) for i in model.zp )
model.calc_spe_zrxi = pyomo.Constraint(rule=calc_spe_zrxi_)

def rotation_speed_min_const_(model):
    return model.zrxi['rotation_speed_rpm']>=500
model.rotation_speed_min_const=pyomo.Constraint(rule= rotation_speed_min_const_)

def rotation_speed_max_const_(model):
    return model.zrxi['rotation_speed_rpm']<=1000
model.rotation_speed_max_const=pyomo.Constraint(rule= rotation_speed_max_const_)

def L_to_S_ratio_min_const_(model):
    return model.zrxi['L_to_S_ratio_pct']>=10
model.L_to_S_ratio_min_const=pyomo.Constraint(rule= L_to_S_ratio_min_const_)

def L_to_S_ratio_max_const_(model):
    return model.zrxi['L_to_S_ratio_pct']<=60
model.L_to_S_ratio_max_const=pyomo.Constraint(rule= L_to_S_ratio_max_const_)

def spe_lessthan_(model):
    return model.spe_zrxi <= model.spe_lim
model.spe_lessthan = pyomo.Constraint(rule=spe_lessthan_)

#Hard constraint on HT2
def ht2_lessthan_(model):
    return model.ht2 <= model.ht2_lim
model.ht2_lessthan = pyomo.Constraint(rule=ht2_lessthan_)

                  
def obj_rule(model):
    return ( ((hr_target  - model.y_hat['Hausner_ratio'])  / model.my['Hausner_ratio'] )**2 
            +((fri_target - model.y_hat['Friability_pct']) / model.my['Friability_pct']  )**2      
            #+ model.ht2/model.ht2_lim   # Penalty (soft) constraint on HT2
            )
model.obj = pyomo.Objective(rule=obj_rule) 

#if solving with couenne locally 
see_solver_diagnostics = False
pyomo.solver = pyomo.SolverFactory('couenne')
results=pyomo.solver.solve(model,tee=see_solver_diagnostics)

#If solving with GAMS locally (you need a GAMS license)
#pyomo.solver = pyomo.SolverFactory('gams:ipopt')
#pyomo.solver.solve(model,tee=see_solver_diagnostics)

#Use these lines to solve with NEOS freebie ! 
#pyomo.solver_manager = pyomo.SolverManagerFactory('neos')
#pyomo.solver_manager.solve(model, opt='ipopt', tee=True)

r = []
for i in model.r:
    r.append(pyomo.value(model.r[i]))  

zrxi = {}
zp=process_var+properties
for i in model.zrxi:
    zrxi[i]=(pyomo.value(model.zrxi[i]))  

y_hat = []
for i in model.y_hat:
    y_hat.append(pyomo.value(model.y_hat[i]))  

tau = []
for i in model.tau:
    tau.append(pyomo.value(model.tau[i]))       
tau   = np.array(tau)
tau   = tau.reshape(-1,1)
print('Process:')
for i in process_var:
    print(i+" :"+str(np.round(zrxi[i],4)))

print('Composition:')
for i,r_val in enumerate(r):
    print(materials[i]+" :"+str(np.round(r_val,4)))
 

print('Product Properties:')
y_var_names=list(Y)[1:]
for i,y_val in enumerate(y_hat):
    print(y_var_names[i]+" :"+str(y_val))
    
plt.figure()
plt.plot( pls_obj['T'][:,0], pls_obj['T'][:,1],'om',label='model')
plt.plot(tau[0],tau[1],'ok',label='Solution found')
plt.axvline(x=0,color='k')
plt.axhline(y=0,color='k')
plt.xlabel('t[1]')
plt.ylabel('t[2]')
plt.legend()
print('\n') 
print('Scores')
for i,t in enumerate(tau):
    print('tnew['+str(i+1)+'] ='+str(t[0]))

print('Solution diagnostics')
print('HT2 Obt: '+str( np.round(pyomo.value(model.ht2),3) )+' vs limit of '+str( np.round(pls_obj['T2_lim95'],3)  ))
print('speX Obt: '+str( np.round(pyomo.value(model.spe_zrxi),3) )+' vs limit of '+str( np.round(pls_obj['speX_lim99'],3)  ))  


