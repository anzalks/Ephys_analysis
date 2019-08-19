#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 23:33:13 2019

@author: anzal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:48:05 2019

@author: anzal
"""

import os
import numpy as np
import neo.io as nio
import matplotlib.pyplot as plt

#set the directory path with files to read
folder_to_read = "/mnt/5D4B-FA71/Data/190814/"
#list out all the files with .mat extension for plotting
for root, dirs, files in os.walk(folder_to_read):
    for file in files:
        if file.endswith(".dat"):
            print(file)
            #import the file of interest
            file_to_read = nio.StimfitIO(root+file)
            segments = ()
            for trace in file_to_read.read_block().segments:
                segments+= (trace,)
                print (segments)
                for segment in segments:
                    for trace in segment.analogsignals:
                        print(trace.sampling_rate)
                        v = trace
                        print(trace)
                        ti = trace.t_start
                        print(trace.t_start)
                        tf = trace.t_stop
                        print(trace.t_stop)
                        tp = float(tf-ti)
                        print(float(tf-ti))
                        sr = float(trace.sampling_rate)
                        print(float(trace.sampling_rate))
                        t = np.linspace(0,tp,int(sr*tp))
                        print(np.linspace(0,tp,int(sr*tp)))
                        plt.plot(t,v)
                        print(t,v)
                        plt.show()
                        
                        
                        
test_file_to_read = "/mnt/5D4B-FA71/Data/190814/c1.dat"
dat_file = nio.RawBinarySignalIO(test_file_to_read)
dat_file



dat_file_stimfit = nio.StimfitIO(test_file_to_read)
dat_file_stimfit
                        