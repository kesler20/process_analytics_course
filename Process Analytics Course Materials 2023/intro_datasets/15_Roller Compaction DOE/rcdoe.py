import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# Load the data from Excel
X    = pd.read_excel('RCDOE.xlsx', 'X', index_col=None, na_values=np.nan)
Y    = pd.read_excel('RCDOE.xlsx', 'Y', index_col=None, na_values=np.nan)

