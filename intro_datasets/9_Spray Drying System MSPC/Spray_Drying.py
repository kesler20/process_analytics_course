import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp

# Load the data from Excel
NOC_data    = pd.read_excel('Spray Drying System.xls', 'NOC',na_values=np.nan)            
Fault_data  = pd.read_excel('Spray Drying System.xls', 'Fault',na_values=np.nan)        
                             
