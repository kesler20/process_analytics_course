import pandas as pd
import numpy as np
import pyphi.pyphi as phi
import pyphi.pyphi_plots as pp

# Load the data from Excel
whiskey_data   = pd.read_excel('Whiskey.xlsx', 'DATA',na_values=np.nan)
whiskey_class  = pd.read_excel('Whiskey.xlsx', 'origin',na_values=np.nan)
pcaobj=phi.pca(whiskey_data,4)


pp.loadings(pcaobj,plotwidth=400)
pp.score_scatter(pcaobj,[1,2],CLASSID=whiskey_class,colorby='Origin',plotwidth=1000)
pp.r2pv(pcaobj)


middleton=[
        'I01',
        'I02',
        'I03',
        'I04',
        'I05',
        'I06',
        'I07',
        'I08',
        'I09',
        'I12']

pp.contributions_plot(pcaobj,whiskey_data,'scores',to_obs=middleton)
