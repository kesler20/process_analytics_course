import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp

# Load the data from Excel
chem_process_data    = pd.read_excel('ChemicalProcessStages.xlsx', 'ProcessData',na_values=np.nan)                                            
chem_process_class   = pd.read_excel('ChemicalProcessStages.xlsx', 'CLASSID',na_values=np.nan)

#Clean the data
chem_process_data,columns_removed = phi.clean_low_variances(chem_process_data)
pcaobj=phi.pca(chem_process_data,3)
pp.r2pv(pcaobj)
pp.score_scatter(pcaobj,[1,2],CLASSID=chem_process_class,colorby='STG9 LABEL (STG9)', 
                 plotwidth=800,plotheight=500,legend_cols=3)
#%%
# #This is a python statement
chem_process_class['Obs ID'][chem_process_class['STG9 LABEL (STG9)']=='CYCLE28']

pp.contributions_plot(pcaobj,chem_process_data,'scores',from_obs=3103,to_obs=3178,lv_space=[1])

myvars=[
        'Temp1 (STG5)',
        'Temp2 (STG5)',
        'Temp3 (STG5)',
        'Inlet Temp (STG9)',
        ]

pp.plot_line_pd(chem_process_data,myvars)

# #Plot the data with with Pandas 
chem_process_data.plot(y=myvars,legend=True,ylabel='Temperature (C)',title='A plot with pandas')
