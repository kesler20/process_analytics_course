# -*- coding: utf-8 -*-
"""
Created on Sun May  1 12:53:38 2022

@author: salva
"""

import pandas as pd
import pyphi as phi
import pyphi_plots as pp
import matplotlib.pyplot as plt

kamyr_process=pd.read_excel('kamyr-digester.xlsx','Process')
kamyr_product=pd.read_excel('kamyr-digester.xlsx','Y')
for i,c in enumerate(kamyr_process.columns):
    if i>0:
        kamyr_process.plot(y=c)

pls_obj=phi.pls(kamyr_process,kamyr_product,5,cross_val=5 )
pp.r2pv(pls_obj)
pp.loadings_map(pls_obj, [1,2])
pp.score_line(pls_obj, 1)
pp.score_line(pls_obj, 2)
pp.score_line(pls_obj, 3)
preds=phi.pls_pred(kamyr_process,pls_obj)
plt.figure()
plt.plot(preds['Yhat'])
plt.title('Y predictions')
