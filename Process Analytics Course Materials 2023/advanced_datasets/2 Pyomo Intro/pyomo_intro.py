# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 22:15:01 2023

@author: salva
"""

from pyomo.environ import *
# These lines are used to send the problem to the NEOS server
import os
os.environ['NEOS_EMAIL'] = 'uchekesla@gmail.com' 


desired_abv = 0.04
desired_vol = 100

model      = ConcreteModel()
model.I    = Set(initialize = ['A','B','W'])
model.vols = Var(model.I,within=NonNegativeReals)
model.abv  = Param(model.I, initialize={'A':0.045,'B':0.037,'W':0.0  })
model.cost = Param(model.I, initialize={'A':0.32 ,'B':0.25 ,'W':0.05 })

def abv_const_(model):
    return desired_abv*desired_vol == sum(model.vols[i]*model.abv[i]  for i in model.I) 
model.abv_const = Constraint(rule=abv_const_)

def vol_const_(model):
    return desired_vol == sum(model.vols[i] for i in model.I)
model.vol_const =  Constraint(rule = vol_const_)

def obj_(model):
    return sum(model.cost[i]*model.vols[i] for i in model.I)
model.obj = Objective(rule=obj_) 

# solver = SolverFactory('gams')
# solver.solve(model,tee=True)

# Use these lines to solve using the NEOS server (open source solver, if you have no solvers in your computer )
solver_manager = SolverManagerFactory('neos')
solver_manager.solve(model,opt='cbc', tee=True)
                

print('Optimal Blend')
for c in ['A','B','W']:
   print('  ', c, ':', model.vols[c](), 'gallons')
print('Volume = ',model.vol_const(), 'gallons')
print('Cost = $', model.obj())