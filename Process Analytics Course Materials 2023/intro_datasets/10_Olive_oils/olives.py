import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp

# Load the data from Excel
olive_oil_data   = pd.read_excel('ItalyOliveOil.xlsx', 'OliveOil',na_values=np.nan)
olive_oil_class  = pd.read_excel('ItalyOliveOil.xlsx', 'Classes',na_values=np.nan)

pcaobj=phi.pca(olive_oil_data,3,cross_val=5)
pp.r2pv(pcaobj)

pp.loadings(pcaobj,plotwidth=400)
pp.score_scatter(pcaobj,[1,2],CLASSID=olive_oil_class,colorby='REGION',plotwidth=1200)
pp.score_scatter(pcaobj,[1,2],CLASSID=olive_oil_class,colorby='SUB-REGION',plotwidth=1200)

# These two lines below are python statements using properties of a pandas dataframe
names_olives_from_north=olive_oil_data['Obs ID'][olive_oil_class['REGION']=='NORTH'].tolist()
names_olives_from_south=olive_oil_data['Obs ID'][olive_oil_class['REGION']=='SOUTH'].tolist()


pp.contributions_plot(pcaobj,olive_oil_data,'scores',to_obs=names_olives_from_north,from_obs=names_olives_from_south)
pp.contributions_plot(pcaobj,olive_oil_data,'scores',to_obs=names_olives_from_south,from_obs=names_olives_from_north)       
 
olive_oil_south= olive_oil_data[olive_oil_class['REGION']=='SOUTH' ]
olive_oil_others= olive_oil_data[olive_oil_class['REGION']!='SOUTH' ]
