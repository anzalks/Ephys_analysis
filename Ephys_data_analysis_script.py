#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 11:12:24 2019

@author: anzal
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import pyabf
import neo 
import stfio

os.chdir('/Users/anzal/Documents/NCBS/UPi_lab work/patch_data/02_01_2018/')


abf = pyabf.ABF("19102001.abf")
abf.setSweep(3)
print(abf.sweepY) # sweep data (ADC)
print(abf.sweepC) # sweep command (DAC)
print(abf.sweepX) # sweep times (seconds)


#visualize each sweep
abf.setSweep(20)
plt.plot(abf.sweepX, abf.sweepY)
plt.show()

#visualize sweeps from range(abf.sweepCount)[from which sweep: till which sweep: in how many steps]
plt.figure(figsize=(8, 5))
for sweepNumber in range(abf.sweepCount)[8:10:1]:
    abf.setSweep(sweepNumber)
    plt.plot(abf.sweepX,abf.sweepY,alpha=.5,label="sweep %d"%(sweepNumber))

plt.legend()
plt.ylabel(abf.sweepLabelY)
plt.xlabel(abf.sweepLabelX)
plt.title("pyABF and Matplotlib are a great pair!")
plt.show()

plt.figure(figsize=(8, 5))

# plot every sweep (with vertical offset)
for sweepNumber in abf.sweepList:
    abf.setSweep(sweepNumber)
    offset = 140*sweepNumber
    plt.plot(abf.sweepX, abf.sweepY+offset, color='C0')

# decorate the plot
plt.gca().get_yaxis().set_visible(False)  # hide Y axis
plt.xlabel(abf.sweepLabelX)
plt.show()


plt.figure(figsize=(8, 5))
for sweepNumber in abf.sweepList:
    abf.setSweep(sweepNumber)
    i1, i2 = 0, int(abf.dataRate * 1) # plot part of the sweep
    dataX = abf.sweepX[i1:i2] + .025 * sweepNumber
    dataY = abf.sweepY[i1:i2] + 15 * sweepNumber
    plt.plot(dataX, dataY, color='C0', alpha=.5)

plt.gca().axis('off') # hide axes to enhance floating effect
plt.show()

x = pd.DataFrame(abf.data)

plt.plot(time_ep, mV_ep)
plt.show()

#
#states = digitalWaveformEpochs[digitalOutputNumber]
#sweepD = np.full(sweepPointCount, 0)
#for epoch in range(len(states)):
#    sweepD[epochPoints[epoch]:epochPoints[epoch+1]] = states[epoch]
#    
#
#neo_obj = neo.io.StimfitIO("19102001.abf")
#print (neo.io.MyFormatIO.mode)
#
#MyFormatIO.supported_objects()
#
#
#
#r = neo.io.AxonIO(filename="19102001.abf")
#bl = r.read_block()
#bl.segments
#print (bl.segments)
#
#print (bl.segments[0].analogsignals)
#
#print (bl.segments[0].eventarrays)
#
#
#
neo_obj = neo.io.stimfitio("19102001.abf")