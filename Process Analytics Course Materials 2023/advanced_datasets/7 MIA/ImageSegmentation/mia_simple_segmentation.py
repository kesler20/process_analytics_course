# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 23:12:00 2023

@author: salva
"""

import imageio.v3 as im
import pyphi as phi
import numpy  as np
import matplotlib.pyplot as plt

#image=im.imread('SGM_0767.jpg')
#image=im.imread('SGM_0007.jpg')
image=im.imread('TwoTablets.JPG')
plt.imshow(image)
for i in np.arange(image.shape[0]):
    slc=image[i]
    if i==0:
        X=slc
    else:
        X=np.vstack((X,slc))
pcaobj=phi.pca(X,2,mcs='center')    
plt.figure()
plt.hist2d(pcaobj['T'][:,0], pcaobj['T'][:,1], bins=(300, 300), cmap=plt.cm.jet)
plt.show()

#%%
reye=im.imread('r_eye.jpg')
leye=im.imread('l_eye.jpg')

for i in np.arange(reye.shape[0]):
    slc=reye[i]
    if i==0:
        Xeye=slc
    else:
        Xeye=np.vstack((Xeye,slc))
for i in np.arange(leye.shape[0]):
    slc=leye[i]
    Xeye=np.vstack((Xeye,slc))
pred=phi.pca_pred(Xeye,pcaobj)
plt.figure()
plt.hist2d(pred['Tnew'][:,0], pred['Tnew'][:,1], bins=(300, 300), cmap=plt.cm.hot)
plt.show()
#%%
#dark skin
#t1box=[83,117]
#t2box=[9,18]

#very light skin
#t1box=[-86,-51]
#t2box=[-7,11]

#Hair and dark
#t1box=[125,250]
#t2box=[-42,20]

#Tablet background
t1box=[-19,117]
t2box=[-15,7]

#Red tablet
t1box=[-163,-36]
t2box=[39,62]

mask=np.zeros((X.shape[0],1))
mask[
(pcaobj['T'][:,0]>t1box[0]) &
(pcaobj['T'][:,0]<t1box[1]) &
(pcaobj['T'][:,1]>t2box[0]) &
(pcaobj['T'][:,1]<t2box[1]) 
]=1
mask=mask.reshape(image.shape[0],image.shape[1])
#mask=np.array([mask,mask,mask])
#maskb=mask.transpose(1,2,0)
plt.figure()
plt.imshow(mask,cmap='gray')
plt.show()

