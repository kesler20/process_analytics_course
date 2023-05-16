#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:07:47 2019

@author: c184156
"""

import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp

# Load the data from Excel
WHO_DATA  = pd.read_excel('WHO_global_burden_disease_death_estimates_sex_age_2008.xlsx', 'DATA',na_values=np.nan)
WHO_DATA,columns_removed  = phi.clean_low_variances(WHO_DATA)
pcaobj=phi.pca(WHO_DATA,1)
pp.loadings(pcaobj,plotwidth=1200)
pp.score_line(pcaobj,1,plotwidth=1200,add_labels=True)











