import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp

# Load the data from Excel
sodp_api_props    = pd.read_excel('SolidOralDrugProduct.xls', 'API_Props', index_col=None, na_values=np.nan)
sodp_tablet_props = pd.read_excel('SolidOralDrugProduct.xls', 'Tablets', index_col=None, na_values=np.nan)

#clean data
sodp_api_props,columns_removed_x = phi.clean_low_variances(sodp_api_props)
sodp_tablet_props,columns_removed_y = phi.clean_low_variances(sodp_tablet_props)

# Build model
plsobj=phi.pls(sodp_api_props,sodp_tablet_props,3,cross_val=5)

plsobj=phi.pls(sodp_api_props,sodp_tablet_props,1)

pp.r2pv(plsobj)

pp.weighted_loadings(plsobj)
