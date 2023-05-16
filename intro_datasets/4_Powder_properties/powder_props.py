# -*- coding: utf-8 -*-
"""
Created on Sun May  1 09:09:26 2022

@author: salva
"""
import pandas as pd
import pyphi as phi
import pyphi_plots as pp

powder_data=pd.read_excel('PharmaPowders.xlsx')
powder_class=powder_data[['Name', 'Class']]
powder_data.drop('Class',axis=1,inplace=True)
pca_obj=phi.pca(powder_data,4)
pp.r2pv(pca_obj)
pp.score_scatter(pca_obj,[1,2],CLASSID=powder_class,colorby='Class')