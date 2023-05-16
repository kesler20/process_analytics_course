# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 19:54:20 2023

@author: salva
"""
import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp
#Load and organize data, make sure rows, columns, all match

jr,materials=phi.parse_materials('FormulatedProductData.xlsx','Materials')
Xraw=[]
for m in materials:
    x_=pd.read_excel( 'FormulatedProductData.xlsx',sheet_name=m)
    Xraw.append(x_)
del x_,m
#Routine to make sure all the rows in X correspond to columns in R
X,jrc=phi.reconcile_rows_to_columns(Xraw, jr)

#Read Process, Quality and Category and make sure rows match
quality=pd.read_excel('FormulatedProductData.xlsx',sheet_name='Quality' )
process=pd.read_excel('FormulatedProductData.xlsx',sheet_name='Process' )
category=pd.read_excel('FormulatedProductData.xlsx',sheet_name='Categorical' )
all_data=jrc.copy()
all_data.append(quality)
all_data.append(process)
all_data.append(category)

all_common_rows= phi.reconcile_rows( all_data)

JR = all_common_rows[:8]
Y  = all_common_rows[8]
Z  = all_common_rows[9]
category=all_common_rows[10]
del all_common_rows,all_data,jr
#%%Show score plots of X spaces to give an idea of how much of the available space is explored 
# Every material's management department should have one of these

i=0
for xr,x in zip(Xraw,X):
    obs=xr[xr.columns[0]].values.tolist()
    obsin=x[x.columns[0]].values.tolist()
    usage=[]
    for o in obs:
        if o in obsin:
            usage.append('Used')
        else:
            usage.append('Not Used')
    tag=pd.DataFrame(np.array([obs,usage]).T,columns=['obsid','usage'])
    pca_obj=phi.pca(xr,2)
    pp.score_scatter(pca_obj, [1,2],CLASSID=tag,colorby='usage',addtitle=materials[i])
    i+=1
del i,o, pca_obj,tag,usage,obs,obsin,xr,x

#%%Build a JR model show ALL plots that can be done
help(phi.jrpls)
#Put all matrices into dictionaries
Xi={}
Ri={}
for m,r,x in zip(materials,JR,X):
    Xi[m]=x
    Ri[m]=r
del m,r,x,X,JR

#%%    
#Start with 5 and show 3 is enough
jrpls_obj=phi.jrpls(Xi, Ri, Y, 3)
pp.r2pv(jrpls_obj,plotwidth=1000,addtitle='JRPLS')
pp.r2pv(jrpls_obj,plotwidth=1000,material='PolymerB',addtitle='JRPLS for Polymer B polymers')

#Analyze what the scores suggest about what is potential problematic material
pp.score_scatter(jrpls_obj, [1,2],CLASSID=category,colorby='QA',addtitle='JRPLS')
pp.score_scatter(jrpls_obj, [1,2],CLASSID=category,colorby='QA',rscores='True',addtitle='JRPLS')
pp.score_scatter(jrpls_obj, [1,2],CLASSID=category,colorby='QA',rscores='True',
                 material='Colorant',add_labels=True,marker_size=12,addtitle='JRPLS')

#Deep dive into the properties with loadings and VIP
pp.loadings_map(jrpls_obj, [1,2])
pp.loadings_map(jrpls_obj, [1,2],textalpha=0.2)
pp.loadings_map(jrpls_obj, [1,2],material='Sugar',addtitle='for Sugar properties')
pp.loadings_map(jrpls_obj, [1,2],material='Colorant',addtitle='for Colorant properties')
pp.loadings_map(jrpls_obj, [1,2],material='SeedPart',addtitle='for SeedPart properties',)
pp.loadings(jrpls_obj)
pp.weighted_loadings(jrpls_obj)
pp.loadings(jrpls_obj,material='SeedPart',addtitle='for SeedPart properties')
pp.vip(jrpls_obj)
pp.vip(jrpls_obj,material='SeedPart')
#%% Add the process data with a TPLS model show ALL plots that can be done
tpls_obj=phi.tpls(Xi,Ri,Z,Y,3)
pp.r2pv(tpls_obj,plotwidth=1000,addtitle='TPLS')
pp.r2pv(tpls_obj,plotwidth=1000,addtitle='TPLS',zspace=True)
pp.score_scatter(tpls_obj, [1,2],CLASSID=category,colorby='QA',addtitle='TPLS')
pp.loadings_map(tpls_obj, [1,2],zspace=True)
pp.loadings_map(tpls_obj, [1,2],zspace=True,textalpha=0.1)
pp.weighted_loadings(tpls_obj,zspace=True)
pp.vip(tpls_obj,zspace=True,plotwidth=1000)

#%%Build a MBPLS contrast scores, loading maps and R2

#Build a dictionary with the multiple blocks of data
#starting with the weighted average physprops per material
RXI={}
for m in materials:
    r      = Ri[m]
    x      = Xi[m]
    rvals  = r.values[:,1:].astype(float)
    xvals  = x.values[:,1:].astype(float)
    rxi_   = rvals@xvals
    rxi_df = pd.DataFrame(rxi_,columns=list(x)[1:])
    rxi_df.insert(0,r.columns[0],r[r.columns[0]].values)
    RXI[m] = rxi_df
#%% need to impute missing data
#Polymer B has missing data, need to impute values from JRPLS 
Pb_props_hat = jrpls_obj['Xhat'][3]  #PolymerB is at index 3
Pb_values    = Xi['PolymerB'].values[:,1:].astype(float)
Pb_values[np.isnan(Pb_values)]=Pb_props_hat[np.isnan(Pb_values)]
newdf        = pd.DataFrame(Pb_values,columns= list(Xi['PolymerB'])[1:])
newdf.insert(0,Xi['PolymerB'].columns[0], Xi['PolymerB'][Xi['PolymerB'].columns[0]]   )
Xi['PolymerB']=newdf

#starting with the weighted average physprops per material
XMB={}
for m in materials:
    r=Ri[m]
    x=Xi[m]
    rvals=r.values[:,1:].astype(float)
    xvals=x.values[:,1:].astype(float)
    rxi_=rvals@xvals
    rxi_df=pd.DataFrame(rxi_,columns=list(x)[1:])
    rxi_df.insert(0,r.columns[0],r[r.columns[0]].values)
    XMB[m]=rxi_df

XMB['Process']=Z
mbpls_obj=phi.mbpls(XMB,Y,3)
pp.score_scatter(mbpls_obj, [1,2],CLASSID=category,colorby='QA')
pp.mb_r2pb(mbpls_obj)
pp.mb_vip(mbpls_obj)
pp.mb_weights(mbpls_obj)
pp.r2pv(mbpls_obj,addtitle='MBPLS',plotwidth=1000)
pp.r2pv(jrpls_obj,addtitle='JRPLS')
pp.r2pv(tpls_obj,addtitle='TPLS')
pp.r2pv(tpls_obj,addtitle='TPLS',zspace=True,plotwidth=1000)