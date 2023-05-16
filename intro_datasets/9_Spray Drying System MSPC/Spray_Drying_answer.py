import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp

# Load the data from Excel
NOC_data    = pd.read_excel('Spray Drying System.xls', 'NOC',na_values=np.nan)            
Fault_data  = pd.read_excel('Spray Drying System.xls', 'Fault',na_values=np.nan)        
                             
# Clean data
NOC_data,columns_removed = phi.clean_low_variances(NOC_data)

#Build a model
pcaobj=phi.pca(NOC_data,3)

#Learn about the process
pp.r2pv(pcaobj)
pp.weighted_loadings(pcaobj)
pp.score_scatter(pcaobj,[1,2])
#%%
# Plot global diagnostics
pp.diagnostics(pcaobj,Xnew=Fault_data,ht2_logscale=True,spe_logscale=True)


#Calculate Contributions to SPE
pp.contributions_plot(pcaobj,Fault_data,'spe',to_obs=312)

#Calculate Contributions to SPE
pp.contributions_plot(pcaobj,Fault_data,'spe',to_obs=341)

#Calculate Contributions to SPE
pp.contributions_plot(pcaobj,Fault_data,'spe',to_obs=395)


# To get names of variables use the instruction:  NOC_data.columns


#Plot the main varible involved
pp.plot_line_pd(Fault_data,'Nozzle Pressure',
             plot_title='Nozzle Pressure',tab_title='Nozzle',xaxis_label='sample')


#Plot multiple variables in one plot
myvars=['Nozzle Pressure',
        'Chamber Inlet Temp',
        'Chamber Outlet Temp'
        ]

pp.plot_line_pd(Fault_data,myvars,
             plot_title='Problematic Variables',tab_title='Fault Diagnosis',xaxis_label='sample')
