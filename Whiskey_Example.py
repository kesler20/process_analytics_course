import pandas as pd
import numpy as np
import pyphi.pyphi as phi
import pyphi.pyphi_plots as pp

# Load the data from Excel
whiskey_data   = pd.read_excel('Whiskey.xlsx', 'DATA',na_values=np.nan)
whiskey_class  = pd.read_excel('Whiskey.xlsx', 'origin',na_values=np.nan)


pcaobj=phi.pca(whiskey_data,2)
pp.loadings(pcaobj)
pp.score_scatter(pcaobj,[1,2],CLASSID=whiskey_class,colorby='Origin',plotheight=400)



