import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp

# Load the data from Excel
food_data    = pd.read_excel('USFOOD.xlsx', 'DATA',na_values=np.nan)                                            
food_type    = pd.read_excel('USFOOD.xlsx', 'TYPES',na_values=np.nan)

#Clean the data
food_data,columns_removed = phi.clean_low_variances(food_data)

pcaobj=phi.pca(food_data,3,cross_val=5)

#%%

pp.diagnostics(pcaobj,score_plot_xydim=[1,2])
pp.r2pv(pcaobj)
pp.loadings(pcaobj)

pp.weighted_loadings(pcaobj,plotwidth=1200)

pp.score_scatter(pcaobj,[1,2],CLASSID=food_type,colorby='TYPE',plotwidth=1200)
pp.score_scatter(pcaobj,[1,3],CLASSID=food_type,colorby='TYPE',plotwidth=1200)
pp.score_scatter(pcaobj,[2,3],CLASSID=food_type,colorby='TYPE',plotwidth=1200)
#%%

pp.contributions_plot(pcaobj,food_data,'scores',to_obs=39,from_obs=88)
pp.contributions_plot(pcaobj,food_data,'scores',to_obs='MCDONALDS_BIG MAC',from_obs='MCDONALDS_ FLT-O-FSH',plotwidth=800)
pp.contributions_plot(pcaobj,food_data,'scores',from_obs='MCDONALDS_BIG MAC',to_obs='MCDONALDS_ FLT-O-FSH',plotwidth=800)

#%% Add lines to do contributions to SPE
pp.contributions_plot(pcaobj,food_data,'spe',to_obs=494) #because python starts from 0
#alternatively
pp.contributions_plot(pcaobj,food_data,'spe',to_obs='SNACKS_SWT POTATO CHIPS_UNSALTED')