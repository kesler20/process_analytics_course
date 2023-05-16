import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp

# Load the data from Excel
olive_oil_data   = pd.read_excel('ItalyOliveOil.xlsx', 'OliveOil',na_values=np.nan)
olive_oil_class  = pd.read_excel('ItalyOliveOil.xlsx', 'Classes',na_values=np.nan)

pcaobj=phi.pca(olive_oil_data,3,cross_val=5)
pp.r2pv(pcaobj)

pp.loadings(pcaobj,plotwidth=400)
pp.score_scatter(pcaobj,[1,2],CLASSID=olive_oil_class,colorby='REGION',plotwidth=1200)
pp.score_scatter(pcaobj,[1,2],CLASSID=olive_oil_class,colorby='SUB-REGION',plotwidth=1200)

#%% Try by region
class_by="REGION"

for c in np.unique(olive_oil_class[class_by]):
    data_train=olive_oil_data[olive_oil_class[class_by]==c]
    data_test=olive_oil_data[olive_oil_class[class_by]!=c]
    pca_obj=phi.pca(data_train,3,shush=True)
    preds=phi.pca_pred(data_test,pca_obj)
    a= preds['speX']<pca_obj['speX_lim99']
    a=a.reshape(-1)
    b= preds['T2']<pca_obj['T2_lim99']
    pass_=np.sum(( a & b )*1)
    print('Using olives from '+ c)
    #print('Passed '+str(pass_)+' out of '+str(data_test.shape[0]))
    print('False passed ratio = '+str(round( 100*pass_/data_test.shape[0],3)))
