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

for root, dirs, files in os.walk(folder_to_read):
    for file in files:
        if file.endswith(".dat"):
            print("Reading ", file)
            #import the file of interest
            file_to_read = nio.RawBinarySignalIO(root+file
                    , sampling_rate=100000
                    , dtype='int16'
                    , nb_channel=1
                    )
            print(file_to_read)
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
          
                    
                    
                    segments[trace_number].analogsignals
                    
                    print(dat_obj)
                    volt = dat_obj.T[0]
                    ti = dat_obj.t_start
                    tf = dat_obj.t_stop
                    tp = float(tf-ti)
                    sr = float(dat_obj.sampling_rate)
                    t = np.linspace(0,tp,int(sr*tp))
                    plt.plot(t,volt)
                    plt.show()
                    
                    
                    dat_obj= trace.analogsignals[trace_number]
                    volt = dat_obj.T[trace_number]
                    ti = dat_obj.t_start
                    tf = dat_obj.t_stop
                    tp = float(tf-ti)
                    sr = float(dat_obj.sampling_rate)
                    t = np.linspace(0,tp,int(sr*tp))
                    plt.plot(t,volt)

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

        

    
