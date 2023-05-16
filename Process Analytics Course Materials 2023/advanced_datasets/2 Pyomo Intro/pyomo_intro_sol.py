# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 22:15:01 2023

@author: salva
"""

from pyomo.environ import *

desired_abv = 0.04
desired_vol = 100

model      = ConcreteModel()
model.I    = Set(initialize = ['A','B','W','C'])
model.vols = Var(model.I,within=NonNegativeReals)
model.abv  = Param(model.I, initialize={'A':0.045,'B':0.037,'W':0.0 , 'C':0.08 })
model.cost = Param(model.I, initialize={'A':0.32 ,'B':0.25 ,'W':0.05, 'C':0.30  })

def abv_const_(model):
    return desired_abv*desired_vol == sum(model.vols[i]*model.abv[i]  for i in model.I) 
model.abv_const = Constraint(rule=abv_const_)

def vol_const_(model):
    return desired_vol == sum(model.vols[i] for i in model.I)
model.vol_const =  Constraint(rule = vol_const_)

#def local_law_(model):
#    return model.vols['W']/sum(model.vols[i] for i in model.I) <= 0.2999
#model.local_law = Constraint(rule = local_law_)

def local_law_(model):
    return model.vols['W'] <= 30
model.local_law = Constraint(rule = local_law_)


def obj_(model):
    return sum(model.cost[i]*model.vols[i] for i in model.I)
model.obj = Objective(rule=obj_) 

solver = SolverFactory('glpk')
solver.solve(model,tee=True)

print('Optimal Blend')
for c in ['A','B','W','C']:
   print('  ', c, ':', model.vols[c](), 'gallons')
print('Volume = ',model.vol_const(), 'gallons')
print('Cost = $', model.obj())