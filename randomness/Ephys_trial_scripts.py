import os
import sys
import pandas as pd
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def find_files(dirname):
    file_list = []
    for root, dirs, files in os.walk(dirname, topdown = False):
       for f in files:
           if f.endswith('.atf'):
               file_list.append(os.path.join(root, f))
    return file_list


# First 9 lines are meta data.
df = pd.read_table(sys.argv[1], skiprows=9)
df = df[int(5e4):int(8e5)]
cols = df.columns
X = df[cols[0]]
Y = df[cols[1]]

# Before finding peak. Threshold the data.
Yy = Y.copy()
Yy[Yy < Yy.mean()] = 0
pI, otherCrap = scipy.signal.find_peaks(Yy)
print(pI)

plt.plot(X, Y)
peakX = X[pI]
plt.plot(peakX, np.ones_like(peakX), 'o')
plt.xlabel(cols[0])
plt.ylabel(cols[1])
plt.show()

