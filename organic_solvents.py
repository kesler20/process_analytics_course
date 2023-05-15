import pandas as pd
import numpy as np
import pyphi.pyphi as phi
import pyphi.pyphi_plots as pp

# Load the data from Excel
solvents_data   = pd.read_excel('SolventProperties.xlsx', 'Solvents',na_values=np.nan)
solvents_class  = pd.read_excel('SolventProperties.xlsx', 'classifiers',na_values=np.nan)
solvents_data,columns_removed =phi.clean_low_variances(solvents_data)
pcaobj = phi.pca(solvents_data,2)
pp.loadings(pcaobj)
pp.score_scatter(pcaobj,[1,2],CLASSID=solvents_class,colorby='Flamability',plotheight=400)
pp.score_scatter(pcaobj,[1,2],CLASSID=solvents_class,colorby='Toxicity',plotheight=400)
pp.score_scatter(pcaobj,[1,2],CLASSID=solvents_class,colorby='Contact Hazzard',plotheight=400)
pp.score_scatter(pcaobj,[1,2],CLASSID=solvents_class,colorby='Others',plotheight=400)