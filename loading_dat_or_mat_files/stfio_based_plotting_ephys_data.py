#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 10:38:42 2019

@author: anzal
"""

import neo
import os
import numpy as np
import matplotlib.pyplot as plt

folder_to_read = os.chdir("/mnt/5D4B-FA71/Data/190814/")

file_to_read = neo.io.NeoMatlabIO(filename='c1.mat')
block_in_file = file_to_read.read_block()
dat_obj = file_to_read.read_block().segments[0].analogsignals[0]
volt = dat_obj.T[0]
ti = dat_obj.t_start
tf = dat_obj.t_stop
tp = float(tf-ti)
sr = float(dat_obj.sampling_rate)
t = np.linspace(0,tp,int(sr*tp))
plt.plot(t,volt)


file_to_read = neo.io.NeoMatlabIO(filename='c1_all.mat')
for obj in file_to_read.read_block().segments:
    dat_obj= obj.analogsignals[0]
    volt = dat_obj.T[0]
    ti = dat_obj.t_start
    tf = dat_obj.t_stop
    tp = float(tf-ti)
    sr = float(dat_obj.sampling_rate)
    t = np.linspace(0,tp,int(sr*tp))
    print(obj)
    plt.plot(t,volt)
    plt.show()
    