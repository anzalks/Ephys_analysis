#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:51:09 2019

@author: anzal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:48:05 2019

@author: anzal
"""
#"name:" inside the segments will give singnal unit/type of recording: use abf.read_block().segments[0].analogsignals[0].name to access it
#channel index will tell which channel it is: abf.read_block().segments[0].analogsignals[0].annotations['channel_index'] (first segment, first signal, channel index)
#the sampling rate is saved as an array in: abf.read_block().segments[0].analogsignals[0].sampling_rate (first segment, first signal, sampling rate, unit is also mentioned next to it)
#str(abf.read_block().segments[0].analogsignals[0].sampling_rate).split()[1] will show the sampling rate units
#str(abf.read_block().segments[0].analogsignals[0][0]).split()[1] will show the unit of singal recorded

#['threshold_index', 'clipped', 'threshold_t', 'threshold_v', 'peak_index', 'peak_t', 'peak_v', 'trough_index', 'trough_t', 'trough_v', 'downstroke_index', 'downstroke', 'downstroke_t', 'downstroke_v', 'upstroke_index', 'upstroke', 'upstroke_t', 'upstroke_v', 'isi_type', 'fast_trough_index', 'fast_trough_t', 'fast_trough_v', 'slow_trough_index', 'slow_trough_t', 'slow_trough_v', 'adp_index', 'adp_t', 'adp_v', 'width', 'upstroke_downstroke_ratio']
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import os
import neo.io as nio
import numpy as np
import matplotlib.pyplot as plt
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor
from scipy.signal import savgol_filter as filter_s

#set the directory path with files to read
folder_to_read = "/home/anzal/Downloads/"
#make a folder to save all the results
try:
    os.mkdir(folder_to_read+'Results')
except:
    pass
try:
    os.mkdir(folder_to_read+'/Results/Current_clamp')
except:
    pass    
#make the file path
results_folder = str(folder_to_read+'/Results/Current_clamp/')
#list out all the files with .dat extension for plotting
for root, dirs, files in os.walk(folder_to_read):
    for file in files:
        if file.endswith(".abf"):
            print(file)
            file_name = str(file.split(".")[0])
            print (file_name)
            #import the file of interest
            file_to_read = nio.StimfitIO(root+file)
            segments = file_to_read.read_block().segments
            #segments = ()
            iteration_number = 0
            segment_number = 0
            Vm_trail = []
            threshold_state = 0
            for segment in segments:
                segment_number +=1            
#                print(segment)
                analog_signals = segment.analogsignals
#                print(analog_signals)
                for trace in analog_signals:
                    iteration_number += 1
#                    print(trace)
                    v = trace
                    v = np.ravel(v)
#                    print(v)
                    if '1.0 pA' == str(v.units):
                        continue
                    if np.isnan(v)[0] == True:
                        continue
                    if iteration_number != 1:
                        traces = 'traces compiled'
                    else:
                         traces = 'trace_alone'
#                    print (Trace_with_features.spike_feature_keys())
                    protocol_type = 'Current_clamp'
                    v = v.magnitude
                    Vm_trace = np.mean(v[len(v)-300:len(v)])
                    tf = trace.t_stop
                    ti = trace.t_start
                    t = np.linspace(0,float(tf - ti), len(v))                    
#                    print(trace.sampling_rate)
                    plt.plot(t,v,label = 'trace numer = '+str(iteration_number))
                    try:
                        Trace_with_features = EphysSweepFeatureExtractor(t=t, v=v, filter = float(trace.sampling_rate)/2500,min_peak=-20.0, dv_cutoff=20.0, max_interval=0.005, min_height=2.0, thresh_frac=0.05, baseline_interval=0.1, baseline_detect_thresh=0.3, id=None)
                        Trace_with_features.process_spikes()
                        neuron_threshold_v = Trace_with_features.spike_feature("threshold_v")
#                        print(threshold_state)
                        if threshold_state == 0 and len(neuron_threshold_v) >=1:
                            neuron_threshold_v = Trace_with_features.spike_feature("threshold_v")[0]
                            neuron_threshold_t = Trace_with_features.spike_feature("threshold_t")[0]
                            plt.plot(neuron_threshold_t,neuron_threshold_v,'o', color ='k',label = 'threshold voltage')
#                            print(neuron_threshold_t,neuron_threshold_v)
                            plt.figtext(1, 0.50, "Neuron's threshold potential = "+ str(np.around(neuron_threshold_v, decimals = 2))+"mV")
                            threshold_state = 1
#                        print(Trace_with_features.filter)
#                        plt.plot(Trace_with_features.spike_feature("peak_t"),Trace_with_features.spike_feature("peak_v"),'r+', label = 'action potentials')
#                        plt.plot([],[],' ',label ='number of peaks = '+ str(len(Trace_with_features.spike_feature("peak_v"))),color = 'red')
                    except:
                        pass
                    plt.title('recording type: '+str(trace.name).split("-")[0]+' '+str(len(segments))+' '+traces+ ' of segment number '+ str(segment_number))
                    plt.ylabel('Amplitude of signal: '+str(trace[0]).split()[1])
                    plt.xlabel('time (S)')
                    plt.ylim(-100,60)
                    plt.legend()
            plt.savefig(results_folder+file_name+" "+"_compiled.png")
            #plt.show()
            plt.close()
            segment_number = 0
            iteration_number = 0
            threshold_state = 0
            for segment in segments:
                segment_number +=1
#                print(segment)
                analog_signals = segment.analogsignals
#                print(analog_signals)
                for trace in analog_signals:
                    iteration_number += 1
                    #print(trace)
                    v = trace
                    v = np.ravel(v)
#                    print(v)
                    if '1.0 pA' == str(v.units):
                        continue
                    if np.isnan(v)[0] == True:
                        continue
                    trace_unit = str(v.units).split('.')[1]
                    protocol_type = 'Current_clamp'
                    v = v.magnitude
                    Vm_i = np.mean(v[0:299])
                    Vm_f = np.mean(v[len(v)-300:len(v)])
                    del_Vm = Vm_f-Vm_i
                    v_diff = v[0:299]-Vm_i
                    print(str(np.mean(v_diff)))
#                    print(del_Vm)
#                    v_filtered = filter_s(v,39, 10)
#                    v1 = np.diff(v_filtered)
##                    print(v1)
#                    tf = trace.t_stop
#                    ti = trace.t_start
#                    t = np.linspace(0,float(tf - ti), len(v))
#                    t_v = np.linspace(0,float(tf - ti), len(v1))
#                    t1 = np.diff(t)[0]
#                    print(t1)
#                    
#                    dv_dt = v1/t1
#                    v2 = np.diff(dv_dt)
#                    d2v_dt2 = v2/t1
#                    v3 = np.diff(d2v_dt2)
#                    t2 = np.linspace(0,float(tf - ti), len(d2v_dt2))
#                    d3v_dt3 = v3/t1
#                    t3 = np.linspace(0,float(tf - ti), len(d3v_dt3))
#                    print(trace.sampling_rate)
                    plt.plot(t,v,label = 'trace numer = '+str(iteration_number))
                    plt.figtext(1, 0.7, 'membrane voltage before : ' + str(np.around(Vm_i,decimals =1))+trace_unit)
                    plt.figtext(1, 0.65, 'membrane voltage after : ' + str(np.around(Vm_f,decimals = 1))+trace_unit)
                    plt.figtext(1, 0.6, 'membrane voltage variation : ' + str(np.around(del_Vm, decimals = 2))+trace_unit)
                    plt.figtext(1, 0.45, 'Number of data points in the trace = ' + str(len(v)))
                    plt.figtext(1, 0.40, 'total time of recording = ' + str(tf-ti))
                    plt.figtext(1, 0.35, 'sampling rate = ' + str(trace.sampling_rate))
                    plt.figtext(1, 0.30, 'sampling rate* total time = ' + str(float(trace.sampling_rate)*(float(tf-ti))))
#                    plt.figtext(1,1, 'membrane voltage: ')
#                    plt.plot(t,v_filtered,label = 'trace numer = '+str(iteration_number))
#                    plt.plot(t_v,dv_dt+10e12,label = 'dv/dt')
#                    plt.plot(t2,d2v_dt2,label = 'd2v/dt2')
#                    plt.plot(t3,d3v_dt3-10e12,label = 'd3v/dt3')
                    try:
                        Trace_with_features = EphysSweepFeatureExtractor(t=t, v=v, filter = float(trace.sampling_rate)/2500,min_peak=-10.0, dv_cutoff=10.0, max_interval=0.005, min_height=2.0, thresh_frac=0.05, baseline_interval=0.1, baseline_detect_thresh=0.3, id=None)
                        Trace_with_features.process_spikes()
                        neuron_threshold_v = Trace_with_features.spike_feature("threshold_v")
#                        print(threshold_state)
                        if threshold_state == 0 and len(neuron_threshold_v) >=1:
                            neuron_threshold_v = Trace_with_features.spike_feature("threshold_v")[0]
                            neuron_threshold_t = Trace_with_features.spike_feature("threshold_t")[0]
                            plt.plot(neuron_threshold_t,neuron_threshold_v,'o', color ='k',label = 'threshold voltage')
#                            print(neuron_threshold_t,neuron_threshold_v)
                            plt.figtext(1, 0.50, "Neuron's threshold potential = "+ str(np.around(neuron_threshold_v, decimals = 2))+"mV")
                            threshold_state = 1
#                        print(Trace_with_features.filter)
                        plt.plot(Trace_with_features.spike_feature("peak_t"),Trace_with_features.spike_feature("peak_v"),'r+', label = 'action potentials')
                        plt.figtext(1, 0.55, 'number of action petentials = '+ str(len(Trace_with_features.spike_feature("peak_v"))))
                    except:
                        pass
                    channel_index = str(trace.annotations['channel_index'])
                    plt.title('recording type: '+str(trace.name).split("-")[0]+' of trace number '+str(iteration_number))
                    plt.ylabel('Amplitude of signal: '+str(trace[0]).split()[1])
                    plt.xlabel('time (S)')
                    plt.ylim(-100,60)
#                    plt.xlim(0.28,0.5)
#                    plt.text(100,100, str(Trace_with_features.spike_feature("width")))
                    plt.legend()
                    plt.savefig(str(results_folder)+str(file_name)+"_"+str(trace.name)+"_"+"from_the_channel_"+channel_index+str(trace.t_start)+".png", bbox_inches="tight")
                   # plt.show()
#                    print(Trace_with_features.spike_feature("threshold_v"))
#                    print(Trace_with_features.spike_feature("peak_v"),Trace_with_features.spike_feature("peak_t"),Trace_with_features.spike_feature("threshold_v"),Trace_with_features.spike_feature("threshold_t"))
#                    print(Trace_with_features.spike_feature("width"))
