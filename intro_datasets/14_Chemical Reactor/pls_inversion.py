import numpy as np
from pyomo.environ import *
import pandas as pd
import pyphi as phi
import pyphi_plots as pp


# Load the data from Excel
process    = pd.read_excel('ChemicalReactor.xls', 'process variables', index_col=None, na_values=np.nan)
product    = pd.read_excel('ChemicalReactor.xls', 'product quality variables', index_col=None, na_values=np.nan)

#clean data
process,columns_removed_x = phi.clean_low_variances(process)
product,columns_removed_y = phi.clean_low_variances(product)

# Build model
plsobj=phi.pls(process,product,2,cross_val=5,cross_val_X=True)

pp.r2pv(plsobj)

pp.weighted_loadings(plsobj)

pp.loadings_map(plsobj,[1,2])

pls_obj = phi.adapt_pls_4_pyomo(plsobj)



see_solver_diagnostics = True

## Enter equality constraints
yeq         = np.array([0, 0, 159000, 0, 0])
yeq_weights = np.array([0, 0,      1, 0, 0])
##

yeq         = phi.np1D2pyomo(yeq)                 #  convert numpy to dictionary
yeq_weights = phi.np1D2pyomo(yeq_weights)         #  convert numpy to dictionary


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
model.var_t       = Param(model.M,initialize = pls_obj['pyo_var_t'])
model.yeq         = Param(model.M, initialize = yeq)
model.yeq_weights = Param(model.M, initialize = yeq_weights)
model.spe_lim    = Param(initialize  = pls_obj['speX_lim95'])

    
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

def calc_spe_x(model):
    return model.spe_x == sum((((model.x[i]-model.mx[i])/model.sx[i] - (model.x_hat[i]-model.mx[i])/model.sx[i])**2) for i in model.N )
model.eq5 = Constraint(rule=calc_spe_x)

def spe_lessthan(model):
    return model.spe_x <= model.spe_lim
model.eq6 = Constraint(rule=spe_lessthan)

#def min_conv(model):
#    return model.y_hat[1]>=.95
#model.eq7 = Constraint(rule=min_conv)

def obj_rule(model):
    return sum(model.yeq_weights[m]*(model.yeq[m]-model.y_hat[m])**2 for m in model.M) + 0.1*model.ht2 + .1*model.spe_x 
model.obj = Objective(rule=obj_rule) 

    
solver = SolverFactory('ipopt')
solver.options['linear_solver']='ma57'
results=solver.solve(model,tee=see_solver_diagnostics)      
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
x_var_names=list(process)
x_var_names=x_var_names[1:]
c=0
for x_val in x:
    print(x_var_names[c]+" :"+str(x_val))
    c=c+1
    
    
y_var_names=list(product)
y_var_names=y_var_names[1:]
c=0
for y_val in y_hat:
    print(y_var_names[c]+" :"+str(y_val))
    c=c+1
print('tau:')
print(tau)