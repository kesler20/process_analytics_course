import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp

# Load the data from Excel
NOC_data    = pd.read_excel('SolventRecoverySystemNOC.xls', 'NOC',na_values=np.nan)            
Fault_data  = pd.read_excel('SolventRecoverySystemNOC.xls', 'Fault',na_values=np.nan)        
                             
# Clean data
NOC_data,columns_removed = phi.clean_low_variances(NOC_data)

#Build a model
pcaobj=phi.pca(NOC_data,5)

#Learn a little bit from it
pp.r2pv(pcaobj)

# Plot global diagnostics
pp.diagnostics(pcaobj,Xnew=Fault_data,ht2_logscale=True,spe_logscale=True)


#Calculate Contributions to SPE
pp.contributions_plot(pcaobj,Fault_data,'spe',to_obs=2400)


#Calculate Contributions
pp.contributions_plot(pcaobj,Fault_data,'scores',to_obs=2400,from_obs=2397)

#Plot the main varible involved
pp.plot_line_pd(Fault_data,'Vacuum Pump Out Pressure PV (PSI)',
             plot_title='Vacuum Pump Out Pressure (PSi)',tab_title='VacPump',xaxis_label='sample')


#Plot multiple variables in one plot
myvars=['E3 Tube Side Inl Pressure PV (PSI)',
        'E2 Temp Control Loop PV (oC)',
        'Vacuum Pump Inl Pressure PV (PSI)',
        'Vacuum Pump Out Pressure PV (PSI)']

pp.plot_line_pd(Fault_data,myvars,
             plot_title='Problematic Variables',tab_title='Fault Diagnosis',xaxis_label='sample')
