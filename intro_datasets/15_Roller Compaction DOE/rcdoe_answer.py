import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# Load the data from Excel
X    = pd.read_excel('RCDOE.xlsx', 'X', index_col=None, na_values=np.nan)
Y    = pd.read_excel('RCDOE.xlsx', 'Y', index_col=None, na_values=np.nan)


# Build model
plsobj=phi.pls(X,Y,6)

#%%

plsobj=phi.pls(X,Y,4)
pp.r2pv(plsobj)
pp.score_scatter(plsobj,[1,2])




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(plsobj['T'][:,0], plsobj['T'][:,1], plsobj['T'][:,2])
ax.set_xlabel('t[1]')
ax.set_ylabel('t[2]')
ax.set_zlabel('t[3]')
plt.show()
