# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import pyphi as phi
import pyphi_plots as pp

eu_food_data=pd.read_excel('Food consumption Europe.xlsx')
classid=pd.read_excel('Food consumption Europe.xlsx',sheet_name='tags')
pca_obj=phi.pca(eu_food_data,3,cross_val=10)
pp.r2pv(pca_obj,plotheight=600)
pp.weighted_loadings(pca_obj)
pp.loadings(pca_obj)
pp.score_scatter(pca_obj, [1,2],CLASSID=classid,colorby='Pop.Group',plotwidth=800)
pp.score_scatter(pca_obj, [1,3],CLASSID=classid,colorby='Pop.Group',plotwidth=800)
pp.loadings_map(pca_obj, [1,3],textalpha=0.2)


#%% Contributions


#leave only adolescents and adults
pp.score_scatter(pca_obj, [1,2],CLASSID=classid,colorby='Pop.Group',plotwidth=800)
to_obs_=[
'Belgium_Adolescents',
'Belgium_Adults',
'Germany_Adolescents',
'Germany_Adults',
'Latvia_Adolescents',
'Latvia_Adults',
'Netherlands_Adolescents',
'Netherlands_Adults',
'Sweden_Adolescents',
'Sweden_Adults',
'United Kingdom_Adolescents',
'United Kingdom_Adults'
]

from_obs_=[
'Austria_Adolescents',
'Austria_Adults',
'Bosnia and Herzegovina_Adolescents',
'Bosnia and Herzegovina_Adults',
'Croatia_Adults',
'Cyprus_Adolescents',
'Cyprus_Adults',
'Czechia_Adolescents',
'Czechia_Adults',
'Denmark_Adolescents',
'Denmark_Adults',
'Estonia_Adolescents',
'Estonia_Adults',
'Finland_Adolescents',
'Finland_Adults',
'France_Adolescents',
'France_Adults',
'Greece_Adolescents',
'Greece_Adults',
'Hungary_Adolescents',
'Hungary_Adults',
'Ireland_Adults',
'Italy_Adolescents',
'Italy_Adults',
'Montenegro_Adolescents',
'Montenegro_Adults',
'Portugal_Adolescents',
'Portugal_Adults',
'Romania_Adolescents',
'Romania_Adults',
'Serbia_Adolescents',
'Serbia_Adults',
'Slovenia_Adolescents',
'Slovenia_Adults',
'Spain_Adolescents',
'Spain_Adults'
    ]

pp.contributions_plot(pca_obj, eu_food_data,'scores',lv_space=[2],from_obs=from_obs_,to_obs=to_obs_)

#%% Outlying observations ?
pp.diagnostics(pca_obj)
eu_food_data=eu_food_data[eu_food_data['obsid']!='Romania_Vegetarians']
eu_food_data=eu_food_data[eu_food_data['obsid']!='Serbia_Vegetarians']
pca_obj=phi.pca(eu_food_data,2,cross_val=10)
pp.r2pv(pca_obj,plotheight=600)
pp.diagnostics(pca_obj)
#%%What is different about the Irish diet ?
pp.contributions_plot(pca_obj, eu_food_data,'spe',to_obs="Ireland_Adults")