# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 19:54:20 2023

@author: salva
"""
import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp
import matplotlib.pyplot as plt
#Load, clean, make sure data matches
jr,materials=phi.parse_materials('PharmaceuticalProduct.xlsx','Materials')
x=[]
for m in materials:
    x_=pd.read_excel( 'PharmaceuticalProduct.xlsx',sheet_name=m)
    x.append(x_)
    
xc,jrc=phi.reconcile_rows_to_columns(x, jr)

quality=pd.read_excel('PharmaceuticalProduct.xlsx',sheet_name='QUALITY')
classid=pd.read_excel('PharmaceuticalProduct.xlsx',sheet_name='IDENTIFIERS')
process=pd.read_excel('PharmaceuticalProduct.xlsx',sheet_name='PROCESS')

jrc.append(process)
jrc.append(quality)
jrc.append(classid)
AUX= phi.reconcile_rows(jrc)

JR_     = AUX[:-3]
process = AUX[-3]
quality = AUX[-2]
classid = AUX[-1]

Ri={}
for j,m in zip(JR_,materials):
    Ri[m]=j
Xi={}
for x_,m in zip(xc,materials):
    Xi[m]=x_

del j,m,x_,JR_,AUX,jrc,jr,xc,x

jrplsobj=phi.jrpls(Xi,Ri,quality,4)

pp.r2pv(jrplsobj,material='MgSt')

pp.loadings( jrplsobj)
pp.loadings( jrplsobj,material='MgSt')

pp.loadings_map( jrplsobj, [1,2],textalpha=0.2)
pp.loadings_map( jrplsobj, [1,2],material='Conf. Sugar')

pp.loadings( jrplsobj)
pp.loadings( jrplsobj,material='Conf. Sugar')

pp.weighted_loadings( jrplsobj)
pp.weighted_loadings( jrplsobj,material='Conf. Sugar')

pp.score_scatter(jrplsobj,[1,2],CLASSID=classid,colorby='QC-OVERALL')
pp.score_scatter(jrplsobj,[1,2],rscores=True)
pp.score_scatter(jrplsobj,[1,2],rscores=True,material='Conf. Sugar')

pp.score_scatter(jrplsobj, [1,2],CLASSID=classid, colorby='QC-30MIN')

pp.vip(jrplsobj,plotwidth=1000 )
pp.vip(jrplsobj,plotwidth=1000 ,material='MgSt')

#Predict a new blend
# L001	A0129	0.557949425	API
# L001	A0130	0.442050575	API
# L001	Lac0003	1	Lactose
# L001	TLC018	1	Talc
# L001	M0012	1	MgSt
# L001	CS0017	1	Conf. Sugar

# L002	A0130	0.309885057	API
# L002	A0131	0.690114943	API
# L002	Lac0004	1	Lactose
# L002	TLC018	1	Talc
# L002	M0012	1	MgSt
# L002	CS0017	1	Conf. Sugar

rnew={
      'API':        [('A0129',0.557949425 ),('A0130',0.442050575 )],
      'Lactose':    [('Lac0003',1)],
      'Talc':       [('TLC018', 1) ],
      'MgSt':       [('M0012',  1)  ],
      'Conf. Sugar':[('CS0017', 1) ]
      }
jrpreds=phi.jrpls_pred(rnew,jrplsobj)
#%%

# Try a TPLS model
tplsobj=phi.tpls(Xi,Ri,process,quality,4)
pp.r2pv(tplsobj)
pp.r2pv(tplsobj,material='API',addtitle='for API')
pp.r2pv(tplsobj,zspace=True)
pp.loadings( tplsobj)
pp.loadings( tplsobj,material='MgSt',addtitle=' for MgSt')
pp.loadings( tplsobj,zspace=True)
pp.weighted_loadings( tplsobj)
pp.weighted_loadings( tplsobj,material='MgSt',addtitle=' for MgSt')
pp.weighted_loadings( tplsobj,zspace=True)
pp.loadings_map(tplsobj,[1,2],plotwidth=800,textalpha=0.2)
pp.loadings_map(tplsobj,[1,2],plotwidth=800,material='MgSt',addtitle=' for MgSt')
pp.loadings_map(tplsobj,[1,2],plotwidth=800,zspace=True,addtitle=' Process Loadings')
pp.score_scatter(tplsobj,[1,2],rscores=True)
pp.score_scatter(tplsobj,[1,2],rscores=True,material='Conf. Sugar',add_labels=True )
pp.score_scatter(tplsobj, [2,3],CLASSID=classid, colorby='QC-30MIN')
pp.vip(tplsobj,plotwidth=1000,zspace=True,addtitle='Z Space')
pp.vip(tplsobj,plotwidth=1000 ,material='MgSt')
pp.vip(tplsobj,plotwidth=1000, addtitle='For all Materials')
#%%
# Predict a new blend
rnew={
      'API':        [('A0129',0.557949425 ),('A0130',0.442050575 )],
      'Lactose':    [('Lac0003',1)],
      'Talc':       [('TLC018', 1) ],
      'MgSt':       [('M0012',  1)  ],
      'Conf. Sugar':[('CS0017', 1) ]
      }
znew=process[process['LotID']=='L001']
znew=znew.values.reshape(-1)[1:].astype(float)
preds=phi.tpls_pred(rnew,znew,tplsobj)
pp.r2pv(tplsobj)
pp.r2pv(tplsobj,zspace=True)


#%%
#Regress the data using Multi-Block PLS
#Calculate MB matrix of weighted average phys props
RXi={'Process':process}
for m in materials:
    r_vals=Ri[m].values[:,1:].astype(float)
    x_vals=Xi[m].values[:,1:].astype(float)
    if np.any(np.isnan(x_vals)):
        xmean=np.tile(phi.mean(x_vals),(x_vals.shape[0],1))
        x_vals[np.isnan(x_vals)]=xmean[np.isnan(x_vals)]
        
    rxi=r_vals@x_vals
    rxi_pd=pd.DataFrame(rxi,columns=m+':'+Xi[m].columns[1:])
    rxi_pd.insert(0,Ri[m].columns[0],Ri[m][Ri[m].columns[0]].values)
    RXi[m]=rxi_pd

mbpls_obj=phi.mbpls(RXi,quality,4,cross_val_=10)
pp.r2pv(mbpls_obj)
pp.loadings(mbpls_obj)
plt.plot(np.cumsum(mbpls_obj['q2Y']),'-o')
pp.predvsobs(mbpls_obj,RXi,quality,CLASSID=classid,colorby='Disso Level')

