## -*- coding: utf-8 -*-
#"""
#Spyder Editor
#
#This is a temporary script file.
#"""
#from neo.io import RawBinarySignalIO
##import neo
##from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor
#import numpy as np
#import matplotlib.pyplot as plt
#
#abf_t1= RawBinarySignalIO("/mnt/5D4B-FA71/Data/anzal/190814")
#
#
#
##abf = neo.io.AxonIO("/Users/anzal/Documents/NCBS/ephys/Work/c1/1.dat")
##abf= neo.io.AxonIO("/Users/anzal/Documents/NCBS/Data/patch_data/02_01_2018/19102000.abf")
#
#dat_obj = abf.read_block().segments[15].analogsignals[0]
##volt = dat_obj.T[0]
###ti = dat_obj.t_start
###tf = dat_obj.t_stop
###tp = float(tf-ti)
###sr = float(dat_obj.sampling_rate)
###t = np.linspace(0,tp,int(sr*tp))
##
##for obj in abf.read_block().segments:
##    dat_obj= obj.analogsignals[0]
##    volt = dat_obj.T[0]
##    ti = dat_obj.t_start
##    tf = dat_obj.t_stop
##    tp = float(tf-ti)
##    sr = float(dat_obj.sampling_rate)
##    t = np.linspace(0,tp,int(sr*tp))
##    plt.plot(t,volt)
##plt.show()
##
##
##i=t
##v = np.array([float(V) for V in volt]) #in mV
##sweep_ext = EphysSweepFeatureExtractor(t=t, v=v, i=i, filter=float(sr)/2500)
##sweep_ext.process_spikes()
##plt.plot(t,v)
##plt.plot(sweep_ext.spike_feature("peak_t"),sweep_ext.spike_feature("peak_v"),'.')
##plt.show()


#impost essential libraries
import os
import scipy.io as sio
import matplotlib.pyplot as plt

#set the directory path with files to read
folder_to_read = "/mnt/5D4B-FA71/Data/190814/"
#list out all the files with .mat extension for plotting
for root, dirs, files in os.walk(folder_to_read):
    for file in files:
        if file.endswith(".mat"):
            #import the file of interest
             file_to_read = sio.loadmat('c1.mat')
             #list out all the keys in dictionary iported from mat file
             traces = list(file_to_read.keys())
             #omit the usual keys which are not traces/time
             traces = traces[3:]
             #loop through each trace/key to extract time and signal
             for trace in traces:
                 #time extrated
                 t=file_to_read[trace][:,0]
                 #signal trace extracted
                 v=file_to_read[trace][:,1]
                 #plots
                 plt.plot(t,v)
                 #plot's shown for each trace
                 plt.show()

##import the file of interest
#file_to_read = scpi.loadmat('c1.mat')
##list out all the keys in dictionary iported from mat file
#traces = list(file_to_read.keys())
##omit the usual keys which are not traces/time
#traces = traces[3:]
##loop through each trace/key to extract time and signal
#for trace in traces:
#    #time extrated
#    t=file_to_read[trace][:,0]
#    #signal trace extracted
#    v=file_to_read[trace][:,1]
#    #plots
#    plt.plot(t,v)
#    #plot's shown for each trace
#    plt.show()















#block_in_file = file_to_read.read_block()
#dat_obj = file_to_read.read_block().segments[0].analogsignals[0]
#volt = dat_obj.T[0]
#ti = dat_obj.t_start
#tf = dat_obj.t_stop
#tp = float(tf-ti)
#sr = float(dat_obj.sampling_rate)
#t = np.linspace(0,tp,int(sr*tp))
#plt.plot(t,volt)
#
#
#
#res = [file_to_read.keys()[i] for i in file_to_read.keys() if i in test_dict]