
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

model.molXsol     = pyomo.Var(model.sol,within=pyomo.NonNegativeReals)
model.molsol      = pyomo.Var(model.sol,within=pyomo.NonNegativeReals)
model.bpsolv      = pyomo.Var(within=pyomo.NonNegativeReals)
model.dens        = pyomo.Param(model.sol, initialize=solvent_densities)
model.mws         = pyomo.Param(model.sol, initialize=solvent_mw)


def sum_unity_(model):
    return sum(model.x[i] for i  in model.mat)==1
model.sum_unity = pyomo.Constraint(rule=sum_unity_)

def calc_scores_(model,i):
    return model.tau[i] == sum(model.Ws[n,i] * ((model.x[n]-model.mx[n])/model.sx[n]) for n in model.N )
model.calc_scores = pyomo.Constraint(model.A,rule=calc_scores_)

def y_hat_calc_(model,i):
    return (model.y_hat[i]-model.my[i])/model.sy[i]==sum(model.Q[i,a]*model.tau[a] for a in model.A)
model.y_hat_calc = pyomo.Constraint(model.M,rule=y_hat_calc_)

def calc_ht2_(model):
    return model.ht2 == sum( model.tau[a]**2/model.var_t[a] for a in model.A)
model.calc_ht2 = pyomo.Constraint(rule=calc_ht2_)

def x_hat_calc_(model,i):
    return (model.x_hat[i]-model.mx[i])/model.sx[i]==sum(model.P[i,a]*model.tau[a] for a in model.A)
model.x_hat_calc = pyomo.Constraint(model.N,rule=x_hat_calc_)

def calc_spe_x_(model):
    return model.spe_x == sum((((model.x[i]-model.mx[i])/model.sx[i] - (model.x_hat[i]-model.mx[i])/model.sx[i])**2) for i in model.N )
model.calc_spe_x = pyomo.Constraint(rule=calc_spe_x_)

#Hard constraint on SPE
def spe_lessthan_(model):
    return model.spe_x <= model.spe_lim
model.spe_lessthan = pyomo.Constraint(rule=spe_lessthan_)

#Hard constraint on HT2
def ht2_lessthan_(model):
    return model.ht2 <= model.ht2_lim
model.ht2_lessthan = pyomo.Constraint(rule=ht2_lessthan_)

def cloud_pt_ub_ (model):
    return model.y_hat['Cloud point [°C]']<=cloud_point_max
model.cloud_pt_ub = pyomo.Constraint(rule = cloud_pt_ub_)           

def cloud_pt_lb_ (model):
    return model.y_hat['Cloud point [°C]']>=cloud_point_min
model.cloud_pt_lb = pyomo.Constraint(rule = cloud_pt_lb_)           

def mn_ub_ (model):
    return model.y_hat['Avg. No. Mol. Wt., Mn [Da]' ]<=mn_max
model.mn_ub = pyomo.Constraint(rule =mn_ub_ )

def calc_mol_solv_(model,i):
    return model.molsol[i]==model.x[i]*model.dens[i]/model.mws[i]
model.calc_mol_solv = pyomo.Constraint(model.sol,rule = calc_mol_solv_ )

def calc_frac_mol_solv_(model,i):
    return model.molXsol[i]==model.molsol[i]/sum(model.molsol[j] for j in model.sol)
model.calc_frac_mol_solv = pyomo.Constraint(model.sol,rule = calc_frac_mol_solv_ )

def min_vc_const_(model):
    return model.x['VC [mol fraction]'] >=0.1
model.min_vc_const = pyomo.Constraint(rule=min_vc_const_)

# equation from ALAMO
def calc_bp_mix_(model):
    return model.bpsolv+273.15 ==( 19.098411547735079096810 * model.x['EtOH BP 78C [mL/mol]']**2 
	                     + 195.26400161690105505841  * model.x['DMF BP 153 C [mL/mol]']**2 
	                     - 107.85449899548922303438  * model.x['DMF BP 153 C [mL/mol]']**3 
	                     + 26.125073067086582057073  * model.x['DMF BP 153 C [mL/mol]']*model.x['THF BP 66 C [mL]'] 
	                     + 335.91101880830291293023)
model.calc_bp_mix = pyomo.Constraint(rule=calc_bp_mix_)

def reac_temp_const_ (model):
    return model.x['Reaction temperature [°C]']<=model.bpsolv
model.reac_temp_const = pyomo.Constraint(rule=reac_temp_const_)       

# maximize the objective function
def obj_rule(model):
    return ( 10*model.x['VC [mol fraction]'] 
            +(model.y_hat['Avg. No. Mol. Wt., Mn [Da]'] / model.my['Avg. No. Mol. Wt., Mn [Da]'] )
            +(model.y_hat['Recovery [%]'] / model.my['Recovery [%]'] )
            + model.ht2/model.ht2_lim  # Penalty (soft) constraint on HT2
            )
model.obj = pyomo.Objective(rule=obj_rule,sense=pyomo.maximize) 

#if solving with couenne locally 
see_solver_diagnostics = True
pyomo.solver = pyomo.SolverFactory('ipopt')
pyomo.solver.options['linear_solver']='ma57'
results=pyomo.solver.solve(model,tee=see_solver_diagnostics)

#If solving with GAMS locally (you need a GAMS license)
#pyomo.solver = pyomo.SolverFactory('gams:ipopt')
#pyomo.solver.solve(model,tee=see_solver_diagnostics)

#Use these lines to solve with NEOS freebie ! 
#pyomo.solver_manager = pyomo.SolverManagerFactory('neos')
#pyomo.solver_manager.solve(model, opt='ipopt', tee=True)


x = []
for i in model.x:
    x.append(pyomo.value(model.x[i]))  

molsol=[]
for i in model.molsol:
    molsol.append(pyomo.value(model.molsol[i]))  

molXsol=[]
for i in model.molsol:
    molXsol.append(pyomo.value(model.molXsol[i]))  
    

y_hat = []
for i in model.y_hat:
    y_hat.append(pyomo.value(model.y_hat[i]))  

tau = []
for i in model.tau:
    tau.append(pyomo.value(model.tau[i]))       
tau   = np.array(tau)
tau   = tau.reshape(-1,1)
print('Materials and Process:')
for i,x_val in enumerate(x):
    print(xvars[i]+" :"+str(np.round(x_val,4)))

print('Boiling Point Mix: '+ str(np.round(pyomo.value(model.bpsolv) ,4) ))

print('Product Properties:')

for i,y_val in enumerate(y_hat):
    print( yvars[i]+" :"+str(y_val))
    
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
print('speX Obt: '+str( np.round(pyomo.value(model.spe_x),3) )+' vs limit of '+str( np.round(pls_obj['speX_lim99'],3)  ))  



