# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 18:17:02 2023

@author: salva
"""

import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_batch as phibatch
import matplotlib.pyplot as plt
import pyphi_plots as pp

data=pd.read_excel('Nylon.xlsx')
phibatch.plot_var_all_batches(data)
pca_obj=phibatch.mpca(data,2)
pp.score_scatter(pca_obj,[1,2])
phibatch.contributions(pca_obj,data,'scores',to_obs=['B50','B51','B52','B53','B54','B55'],dyn_conts=True)

Xuf,v,b=phibatch.unfold_horizontal(data)
pca_obj_R=phi.varimax_rotation(pca_obj ,Xuf)
pp.score_scatter(pca_obj_R,[1,2])
phibatch.loadings_abs_integral(pca_obj_R)

phibatch.plot_batch(data,['B50','B51','B52','B53','B54','B55'],which_var=['Var 7'],include_set=True,single_plot=True)
phibatch.plot_batch(data,['B50','B51','B52','B53','B54','B55'],which_var=['Var 8'],include_set=True,single_plot=True)
phibatch.plot_batch(data,['B50','B51','B52','B53','B54','B55'],which_var=['Var 9'],include_set=True,single_plot=True)