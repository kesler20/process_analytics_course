import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp


# Load the data from Excel
passengers   = pd.read_excel('Titanic_DA.xlsx', 'PASSENGER', index_col=None, na_values=np.nan)
survived     = pd.read_excel('Titanic_DA.xlsx', 'SURVIVED', index_col=None, na_values=np.nan)
 
plsobj=phi.pls(passengers,survived,1,force_nipals=True)

pp.loadings(plsobj)