#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:51:09 2019

@author: anzal
"""


#"name:" inside the segments will give singnal unit/type of recording: use abf.read_block().segments[0].analogsignals[0].name to access it
#channel index will tell which channel it is: abf.read_block().segments[0].analogsignals[0].annotations['channel_index'] (first segment, first signal, channel index)
#the sampling rate is saved as an array in: abf.read_block().segments[0].analogsignals[0].sampling_rate (first segment, first signal, sampling rate, unit is also mentioned next to it)
#str(abf.read_block().segments[0].analogsignals[0].sampling_rate).split()[1] will show the sampling rate units
#str(abf.read_block().segments[0].analogsignals[0][0]).split()[1] will show the unit of singal recorded

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning) 
import os
import neo.io as nio
import numpy as np
import matplotlib.pyplot as plt
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor

#set the directory path with files to read
folder_to_read = "/home/anzal/Downloads/"
#make a folder to save all the results
try:
    os.mkdir(folder_to_read+'Results')
except:
    pass
try:
    os.mkdir(folder_to_read+'/Results/Voltage_clamp')
except:
    pass
#make the file path
results_folder = str(folder_to_read+'/Results/Voltage_clamp/')
#list out all the files with .dat extension for plotting
for root, dirs, files in os.walk(folder_to_read):
    for file in files:
        if file.endswith(".abf"):
#            print(file)
            file_name = str(file).split(".")[0]
            print (file_name)
            #import the file of interest
            file_to_read = nio.AxonIO(root+file)
            segments = file_to_read.read_block().segments
            #segments = ()
            iteration_number = 0
            for segment in segments:
#                print(segment)
                analog_signals = segment.analogsignals
#                print(analog_signals)
                for trace in analog_signals:
                    iteration_number += 1
#                    print(trace)
                    v = trace
                    v = np.ravel(v)
#                    print(v)
                    if '1.0 mV' == str(v.units):
                        continue
                    if np.isnan(v)[0] == True:
                        continue
                    if iteration_number != 1:
                        traces = 'traces_compiled'
                    else:
                         traces = 'trace alone'
                    protocol_type = 'Voltage_clamp'
                    v = v.magnitude
                    tf = trace.t_stop
                    ti = trace.t_start
                    t = np.linspace(0,float(tf - ti), len(v))
#                    print(trace.sampling_rate)
                    plt.plot(t,v,label = 'trace numer = '+str(iteration_number))
                    try:
                        Trace_with_features = EphysSweepFeatureExtractor(t=t, v=v, filter = float(trace.sampling_rate)/2500,min_peak=-20.0, dv_cutoff=20.0, max_interval=0.005, min_height=2.0, thresh_frac=0.05, baseline_interval=0.1, baseline_detect_thresh=0.3, id=None)
                        Trace_with_features.process_spikes()
#                        print(Trace_with_features.filter)
#                        plt.plot(Trace_with_features.spike_feature("peak_t"),Trace_with_features.spike_feature("peak_v"),'r+', label = 'action potentials')
#                        plt.plot([],[],' ',label ='number of peaks = '+ str(len(Trace_with_features.spike_feature("peak_v"))),color = 'red')
                    except:
                        pass
                    plt.title('recording type: '+ protocol_type +' '+str(len(segments))+' '+traces)
                    plt.ylabel('Amplitude of signal: '+str(trace[0]).split()[1])
                    plt.xlabel('time (S)')
                    plt.ylim(-500,100)
                    plt.legend()
            plt.savefig(results_folder+file_name+" "+'_compiled.png')
            plt.show()
            plt.close()
            for segment in segments:
#                print(segment)
                analog_signals = segment.analogsignals
#                print(analog_signals)
                for trace in analog_signals:
                    iteration_number += 1
                    #print(trace)
                    v = trace
                    v = np.ravel(v)
#                    print(v)
                    if '1.0 mV' == str(v.units):
                        continue
                    if np.isnan(v)[0] == True:
                        continue
                    protocol_type = 'Voltage_clamp'
                    v = v.magnitude
                    tf = trace.t_stop
                    ti = trace.t_start
                    t = np.linspace(0,float(tf - ti), len(v))
#                    print(trace.sampling_rate)
                    plt.plot(t,v,label = str(trace.name).split("-")[0])
#                    try:
#                        Trace_with_features = EphysSweepFeatureExtractor(t=t, v=v, filter = float(trace.sampling_rate)/2500,min_peak=-20.0, dv_cutoff=20.0, max_interval=0.005, min_height=2.0, thresh_frac=0.05, baseline_interval=0.1, baseline_detect_thresh=0.3, id=None)
#                        Trace_with_features.process_spikes()
##                        print(Trace_with_features.filter)
#                        plt.plot(Trace_with_features.spike_feature("peak_t"),Trace_with_features.spike_feature("peak_v"),'r+', label = 'action potentials')
#                        plt.plot([],[],' ',label ='number of action petentials = '+ str(len(Trace_with_features.spike_feature("peak_v"))),color = 'red')
#                    except:
#                        pass
                    channel_index = str(trace.annotations['channel_index'])
                    plt.title('recording type: '+str(trace.name).split("-")[0]+' of trace number '+str(iteration_number))
                    plt.ylabel('Amplitude of signal: '+str(trace[0]).split()[1])
                    plt.xlabel('time (S)')
                    plt.ylim(-500,100)
                    plt.legend()
                    plt.savefig(str(results_folder)+str(file_name)+"_"+str(trace.name)+"_"+"from_the_channel_"+channel_index+str(trace.t_start)+".png")
                    plt.show()