import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp

# Load the data from Excel
sodp_api_props    = pd.read_excel('SolidOralDrugProduct.xls', 'API_Props', index_col=None, na_values=np.nan)
sodp_tablet_props = pd.read_excel('SolidOralDrugProduct.xls', 'Tablets', index_col=None, na_values=np.nan)

