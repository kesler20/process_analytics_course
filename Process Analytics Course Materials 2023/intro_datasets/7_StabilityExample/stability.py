import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp

# Load the data from Excel
stability_data    = pd.read_excel('Granulation Screening.xls', 'StabilityBLCorrected',na_values=np.nan)
stability_timing  = pd.read_excel('Granulation Screening.xls', 'Class',na_values=np.nan)

stability_data,columns_removed = phi.clean_low_variances(stability_data)

pcaobj=phi.pca(stability_data,1,mcs=False)
pp.score_line(pcaobj,1,plotwidth=1200,add_labels=True,CLASSID=stability_timing,colorby='Blend')
pp.score_line(pcaobj,1,plotwidth=1200,add_labels=True,CLASSID=stability_timing,colorby='Sample')