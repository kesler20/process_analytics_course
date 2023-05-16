import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp

# Load the data from Excel
process    = pd.read_excel('ChemicalReactor.xls', 'process variables', index_col=None, na_values=np.nan)
product    = pd.read_excel('ChemicalReactor.xls', 'product quality variables', index_col=None, na_values=np.nan)

#clean data
process,columns_removed_x = phi.clean_low_variances(process)
product,columns_removed_y = phi.clean_low_variances(product)

# Build model
plsobj=phi.pls(process,product,3,cross_val=5)

pp.r2pv(plsobj)

plsobj=phi.pls(process,product,2,cross_val=5)

pp.r2pv(plsobj)

pp.weighted_loadings(plsobj)

pp.loadings_map(plsobj,[1,2])
