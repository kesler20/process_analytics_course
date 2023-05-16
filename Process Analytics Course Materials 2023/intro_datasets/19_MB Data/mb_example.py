# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 18:40:39 2022

@author: salva
"""

import pyphi as phi
import pyphi_plots as pp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

api_data=pd.read_excel('MB with Dissolution.xlsx','API')
lac_data=pd.read_excel('MB with Dissolution.xlsx','LAC')
mst_data=pd.read_excel('MB with Dissolution.xlsx','MST')
cap_data=pd.read_excel('MB with Dissolution.xlsx','CAP')
gran_data=pd.read_excel('MB with Dissolution.xlsx','Gran')
drying_data=pd.read_excel('MB with Dissolution.xlsx','DRYING')
cqa_data=pd.read_excel('MB with Dissolution.xlsx','CQA')

mbdata={'API':api_data,
        'LAC':lac_data,
        'MST':mst_data,
        'CAP':cap_data,
        'Gran':gran_data,
        'DRYING':drying_data
        }


mbpls_obj=phi.mbpls(mbdata,cqa_data,2)
preds=phi.pls_pred(mbdata, mbpls_obj)

pp.r2pv(mbpls_obj)
pp.mb_r2pb(mbpls_obj)
pp.mb_weights(mbpls_obj)
pp.mb_vip(mbpls_obj)


