# -*- coding: utf-8 -*-
"""
Created on Wed May 10 19:17:11 2023

@author: salva
"""

import pyphi as phi
import pandas as pd
import pyphi_plots as pp

data= pd.read_excel('hybrid_data.xlsx')

y_data=data[['RUN #','CQA1', 'CQA2']]
       
x_data=data[['RUN #', 'Inlet_blend_bulk_density', 'Ribbon_density',
              'Ribbon_porosity', 'Roll_gap_width', 'Gap_set_point', 'Roll_Force']]

#Build a PLS model for our CQA's
pls_obj=phi.pls(x_data,y_data,2)
pp.r2pv(pls_obj)
pp.score_scatter(pls_obj,[1,2],add_labels=True)
pls_rot=phi.varimax_rotation(pls_obj, x_data,Y=y_data)
pp.predvsobs(pls_obj, x_data, y_data)
pp.r2pv(pls_rot)
pp.score_scatter(pls_rot,[1,2],add_labels=True)
phi.export_2_gproms(pls_obj)