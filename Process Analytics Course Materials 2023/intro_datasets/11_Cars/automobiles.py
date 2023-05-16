#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script to build a PCA and a PLS model
"""

import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp

# Load the data from Excel
Cars_Features    = pd.read_excel('Automobiles PLS.xls', 'Features', index_col=None, na_values=np.nan)
Cars_Performance = pd.read_excel('Automobiles PLS.xls', 'Performance', index_col=None, na_values=np.nan)
Cars_CLASSID     = pd.read_excel('Automobiles PLS.xls', 'CLASSID', index_col=None, na_values=np.nan)

Cars_Features,columns_removed_x = phi.clean_low_variances(Cars_Features)
Cars_Performance,columns_removed_y = phi.clean_low_variances(Cars_Performance)


# Build a PLS model with 3 PC's, cross validating by elements removing 5% of the data per round
plsobj=phi.pls(Cars_Features,Cars_Performance,2,cross_val=5)
# Build a PLS model with 3 PC's, cross validating by elements removing 5% of the data per round add crossval of X Space
plsobj=phi.pls(Cars_Features,Cars_Performance,2,cross_val=5,cross_val_X=True)

#%%
#Make some plots 
pp.r2pv(plsobj)
pp.loadings(plsobj)
pp.score_scatter(plsobj,[1,2],CLASSID=Cars_CLASSID,colorby='Origin',add_ci=True)
pp.score_scatter(plsobj,[1,2],CLASSID=Cars_CLASSID,colorby='Cylinders')


pp.weighted_loadings(plsobj)
pp.vip(plsobj)
pp.loadings_map(plsobj,[1,2])


pp.contributions_plot(plsobj,Cars_Features,'scores',to_obs='Car1',from_obs='Car4')

pp.predvsobs(plsobj,Cars_Features,Cars_Performance)
pp.predvsobs(plsobj,Cars_Features,Cars_Performance,CLASSID=Cars_CLASSID,colorby='Origin')
pp.predvsobs(plsobj,Cars_Features,Cars_Performance,CLASSID=Cars_CLASSID,colorby='Origin',x_space=True)


