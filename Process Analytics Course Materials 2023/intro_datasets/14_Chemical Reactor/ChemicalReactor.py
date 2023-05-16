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
