import pyomo.environ as pyomo_env
import os
os.environ['NEOS_EMAIL'] = 'uchekesla@gmail.com' 

desired_abv = 0.04
desired_vol = 100

model      = pyomo_env.ConcreteModel()
model.I    = pyomo_env.Set(initialize = ['A','B','W',"C"])
model.vols = pyomo_env.Var(model.I,within=pyomo_env.NonNegativeReals)
model.abv  = pyomo_env.Param(model.I, initialize={'A':0.045,'B':0.037,'W':0.0,"C": 0.08})
model.cost = pyomo_env.Param(model.I, initialize={'A':0.32 ,'B':0.25 ,'W':0.05,"C": 0.3})

def abv_const_(model):
    return desired_abv*desired_vol == sum(model.vols[i]*model.abv[i]  for i in model.I) 
model.abv_const = pyomo_env.Constraint(rule=abv_const_)

def vol_const_(model):
    return desired_vol == sum(model.vols[i] for i in model.I)
model.vol_const =  pyomo_env.Constraint(rule = vol_const_)

def obj_(model):
    return sum(model.cost[i]*model.vols[i] for i in model.I)
model.obj = pyomo_env.Objective(rule=obj_) 

# Use these lines to solve using the NEOS server (open source solver, if you have no solvers in your computer )
solver_manager = pyomo_env.SolverManagerFactory('neos')
solver_manager.solve(model,opt='cbc', tee=True)
                

print('Optimal Blend')
for c in ['A','B','W',"C"]:
   print('  ', c, ':', model.vols[c](), 'gallons')
print('Volume = ',model.vol_const(), 'gallons')
print('Cost = $', model.obj())