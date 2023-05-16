import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp
import matplotlib.pyplot as plt
import pyomo.environ as pyomo
import os

# Feel free to use your email
os.environ['NEOS_EMAIL'] = 'pyphisoftware@gmail.com' 

#This is how to fit a basic LPLS model and do some plots
X=pd.read_excel('PharmaceuticalMaterials.xlsx',sheet_name='Materials')
R=pd.read_excel('PharmaceuticalMaterials.xlsx',sheet_name='Blends')
Y=pd.read_excel('PharmaceuticalMaterials.xlsx',sheet_name='Quality')
mat_class=pd.read_excel('PharmaceuticalMaterials.xlsx',sheet_name='material_classes')
Rnew=pd.read_excel('PharmaceuticalMaterials.xlsx',sheet_name='ans')

#show with 7 PC's sharp drop after 5th
lpls_obj = phi.lpls(X,R,Y,5)

#Create a dataframe with Xhat
xhat=pd.DataFrame(lpls_obj['Xhat'],columns=X.columns[1:])
xhat.insert(0,X.columns[0],X[X.columns[0]])
  
#Impute missing data in X
Xvals=X.values[:,1:].astype(float)
Xvals[np.isnan(Xvals)]=lpls_obj['Xhat'][np.isnan(Xvals)]

#Create RXI
Rvals=R.values[:,1:].astype(float)
#Calculate weighted average physprops and build DataFrame
RXi=Rvals @ Xvals

RXi_df=pd.DataFrame(RXi,columns=X.columns[1:])
RXi_df.insert(0,R.columns[0],R[R.columns[0]])
#Build PLS
pls_obj=phi.pls(RXi_df,Y,5,cross_val=10)

#%%USE PLS IN FORMULATION EXAMPLE
vicqa_target = 21.9
cqa_target = 98.8
kqa_min=9
#1st scenario
icqa_target = 5.46
#2nd scenario
#icqa_target = 2.27

#Condition matrices as dictionaries
pls_obj = phi.adapt_pls_4_pyomo(pls_obj,use_var_ids=True)
materials  = list(R)[1:]
properties = list(X)[1:]
Xpyo       = dict(((materials[i],properties[j]), Xvals[i][j]) for i in range(Xvals.shape[0]) for j in range(Xvals.shape[1]))

see_solver_diagnostics = True
model             = pyomo.ConcreteModel()
model.A           = pyomo.Set(initialize = pls_obj['pyo_A'] )
model.p           = pyomo.Set(initialize = pls_obj['pyo_N'] ) #Mat Properties Index
model.N           = pyomo.Set(initialize = pls_obj['pyo_M'] ) #  Y Properties Index
model.M           = pyomo.Set(initialize = materials)       #Mat Index
model.y_hat       = pyomo.Var(model.N, within=pyomo.Reals)
model.rxi_hat     = pyomo.Var(model.p, within=pyomo.Reals)
model.rxi         = pyomo.Var(model.p, within=pyomo.Reals) #Weighted average phys-props
model.r           = pyomo.Var(model.M, within=pyomo.NonNegativeReals) #blending info
model.tau         = pyomo.Var(model.A,within =pyomo.Reals)
model.spe_rxi     = pyomo.Var(within = pyomo.Reals)
model.ht2         = pyomo.Var(within = pyomo.Reals)
model.Ws          = pyomo.Param(model.p,model.A,initialize = pls_obj['pyo_Ws'])
model.P           = pyomo.Param(model.p,model.A,initialize = pls_obj['pyo_P'])
model.Q           = pyomo.Param(model.N,model.A,initialize = pls_obj['pyo_Q'])
model.mx          = pyomo.Param(model.p,initialize = pls_obj['pyo_mx'])
model.sx          = pyomo.Param(model.p,initialize = pls_obj['pyo_sx'])
model.my          = pyomo.Param(model.N,initialize = pls_obj['pyo_my'])
model.sy          = pyomo.Param(model.N,initialize = pls_obj['pyo_sy'])
model.var_t       = pyomo.Param(model.A,initialize = pls_obj['pyo_var_t'])
model.spe_lim    = pyomo.Param(initialize  = pls_obj['speX_lim95'])
model.ht2_lim    = pyomo.Param(initialize  = pls_obj['T2_lim95'])
model.X          = pyomo.Param(model.M,model.p,initialize  = Xpyo)
#Adding Binaries to control number of materials
model.rbin       = pyomo.Var(model.M ,within=pyomo.Binary)

#Binary constraints so that only API 5 is used, and is 60% of formula
def binary_constraint(model,i):
        return model.r[i]<=model.rbin[i]
model.bincon1 = pyomo.Constraint(model.M,rule=binary_constraint)

def api5_con1_(model):
    return model.rbin['API 5']==1
model.bincon2 = pyomo.Constraint(rule=api5_con1_)

def api5_conc_(model):
    return model.r['API 5']==0.6
model.api5_conc = pyomo.Constraint(rule=api5_conc_)

other_apis=['API 1','API 2','API 3','API 4','API 6','API 7','API 8','API 9']
model.oa = pyomo.Set (initialize=other_apis)

def other_apis_bincon_(model,i):
    return model.rbin[i]==0.0
model.bincon3 = pyomo.Constraint(model.oa,rule=other_apis_bincon_)


def sum_unity_(model):
    return sum(model.r[i] for i  in model.M)==1
model.con0 = pyomo.Constraint(rule=sum_unity_)

def calc_rxi_(model,i):
    return model.rxi[i] == sum(model.X[m,i]*model.r[m]  for m in model.M)
model.con1= pyomo.Constraint(model.p,rule=calc_rxi_ )

def calc_scores(model,i):
    return model.tau[i] == sum(model.Ws[n,i] * ((model.rxi[n]-model.mx[n])/model.sx[n]) for n in model.p )
model.con2 = pyomo.Constraint(model.A,rule=calc_scores)

def y_hat_calc(model,i):
    return (model.y_hat[i]-model.my[i])/model.sy[i]==sum(model.Q[i,a]*model.tau[a] for a in model.A)
model.con3 = pyomo.Constraint(model.N,rule=y_hat_calc)

def calc_ht2(model):
    return model.ht2 == sum( model.tau[a]**2/model.var_t[a] for a in model.A)
model.con4 = pyomo.Constraint(rule=calc_ht2)

def rxi_hat_calc(model,i):
    return (model.rxi_hat[i]-model.mx[i])/model.sx[i]==sum(model.P[i,a]*model.tau[a] for a in model.A)
model.con5 = pyomo.Constraint(model.p,rule=rxi_hat_calc)

def calc_spe_rxi(model):
    return model.spe_rxi == sum((((model.rxi[i]-model.mx[i])/model.sx[i] - (model.rxi_hat[i]-model.mx[i])/model.sx[i])**2) for i in model.p )
model.con6 = pyomo.Constraint(rule=calc_spe_rxi)

def spe_lessthan(model):
    return model.spe_rxi <= model.spe_lim
model.con7 = pyomo.Constraint(rule=spe_lessthan)

def ht2_lessthan(model):
    return model.ht2 <= model.ht2_lim
model.con8 = pyomo.Constraint(rule=ht2_lessthan)

def kqa_gt_(model):
    return model.y_hat['KQA']>=kqa_min
model.con9 = pyomo.Constraint(rule =kqa_gt_ )
                      
def obj_rule(model):
    return ( ((vicqa_target- model.y_hat['VICQA']) / model.my['VICQA'] )**2 
            +((icqa_target - model.y_hat['ICQA'] ) / model.my['ICQA']  )**2 
            +((cqa_target  - model.y_hat['CQA']  ) / model.my['CQA']   )**2            
            )
model.obj = pyomo.Objective(rule=obj_rule) 

#if solving with couenne locally 
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

rxi = []
for i in model.rxi:
    rxi.append(pyomo.value(model.rxi[i]))  

y_hat = []
for i in model.y_hat:
    y_hat.append(pyomo.value(model.y_hat[i]))  

tau = []
for i in model.tau:
    tau.append(pyomo.value(model.tau[i]))       
tau   = np.array(tau)
tau   = tau.reshape(-1,1)

print('Composition:')
for i,r_val in enumerate(r):
    print(materials[i]+" :"+str(np.round(r_val,4)))
print('\n')    

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
print('speX Obt: '+str( np.round(pyomo.value(model.spe_rxi),3) )+' vs limit of '+str( np.round(pls_obj['speX_lim99'],3)  ))  



