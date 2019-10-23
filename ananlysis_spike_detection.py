#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:58:52 2019

@author: anzal
"""

import os
import neo.io as nio
import numpy as np
import matplotlib.pyplot as plt
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor

#Trace_with_features.spike_feature_keys()


file_to_read = nio.StimfitIO('/mnt/5D4B-FA71/Data/190822/trace_alone.dat')
segments = file_to_read.read_block().segments
iteration_number = 0
for segment in segments:
    #print(segment)
    analog_signals = segment.analogsignals
    #print(analog_signals)
    for trace in analog_signals:
        iteration_number += 1
        #print(trace)
        sampling_rate = trace.sampling_rate
        #print(sampling_rate)
        v = trace
        v = np.ravel(v)
        #pick only traces with current clamp data
        if '1.0 pA' == str(v.units):
            continue
        if np.isnan(v)[0] == True:
            continue
        #print(v)
        #print ('ttttttttttt')
        v = v.magnitude
        #print(v)
        tf = trace.t_stop
        #print(tf)
        ti = trace.t_start
        #print(ti)
        t = np.linspace(0,float(tf - ti), len(v))
        #print(t)
        #i = np.zeros(len(v))
        #print(np.isnan(v)[0])
        #print('##########')
        print(trace.sampling_rate)
        plt.plot(t,v,label = (iteration_number))
        try:
            Trace_with_features = EphysSweepFeatureExtractor(t=t, v=v, filter = float(trace.sampling_rate)/2500,min_peak=-30.0, dv_cutoff=20.0, max_interval=0.005, min_height=2.0, thresh_frac=0.05, baseline_interval=0.1, baseline_detect_thresh=0.3, id=None)
            Trace_with_features.process_spikes()
            print(Trace_with_features.filter)        
            plt.plot(Trace_with_features.spike_feature("peak_t"),Trace_with_features.spike_feature("peak_v"),'k+')
        except:
            pass
        plt.title('recording type: '+str(trace.name).split("-")[0]+' '+str(len(analog_signals))+' '+'traces'+' '+'_compiled')
        plt.ylabel('Amplitude of signal: '+str(trace[0]).split()[1])
        plt.xlabel('time (mS)')
        plt.legend()
        plt.show()
        
        
        
file_to_read = nio.StimfitIO('/mnt/5D4B-FA71/Data/190822/trace_alone.dat')
segments = file_to_read.read_block().segments
analog_signal = segment.analogsignals
trace_1 = segment.tr
