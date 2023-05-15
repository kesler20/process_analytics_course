import pandas as pd
import numpy as np
import pyphi.pyphi as phi
import pyphi.pyphi_plots as pp

# Load the data from Excel
chem_process_data    = pd.read_excel('ChemicalProcessStages.xlsx', 'ProcessData',na_values=np.nan)                                            
chem_process_class   = pd.read_excel('ChemicalProcessStages.xlsx', 'CLASSID',na_values=np.nan)

#Clean the data
chem_process_data,columns_removed = phi.clean_low_variances(chem_process_data)
#Build the model
pcaobj=phi.pca(chem_process_data,3)
