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

 
import os
import neo.io as nio
#import numpy as np
import matplotlib.pyplot as plt

#set the directory path with files to read
folder_to_read = "/mnt/5D4B-FA71/Data/190924/"
#make a folder to save all the results
#os.mkdir(folder_to_read+'Results/')
#make the file path
results_folder = str(folder_to_read+'Results/')
#list out all the files with .dat extension for plotting
for root, dirs, files in os.walk(folder_to_read):
    for file in files:
        if file.endswith(".dat"):
            #print(file)
            file_name = str(file).split(".")[0]
            print (file_name)
            #import the file of interest
            file_to_read = nio.StimfitIO(root+file)
            segments = file_to_read.read_block().segments
            #segments = ()
            for segment in segments:
                print(segment)
                analog_signals = segment.analogsignals
                print(analog_signals)
                for trace in analog_signals:
                    print(trace)
#                    for trace_no in trace:
#                        v = trace_no
#                        print(v)
                    v = trace
                    print(v)
#                    if np.isnan(trace).all() == np.nan:
#                       break
#                       v = trace
                    plt.plot(v)
                    plt.title('recording type: '+str(trace.name).split("-")[0]+' '+str(len(analog_signals))+' '+'traces'+' '+'_compiled')
                    plt.ylabel('Amplitude of signal: '+str(trace[0]).split()[1])
                    plt.xlabel('time (mS)')
                    #plt.tight_layout()
                    plt.legend()
            #save the figure in the results folder with png as extension, fiile name, Key detail(trace number)     
            plt.savefig(results_folder+file_name+" "+'_compiled.png')
            plt.show()
            plt.close()
            for segment in segments:
                print(segment)
                analog_signals = segment.analogsignals
                #print(analog_signals)
                for trace in analog_signals:
                    v = trace
                    channel_index = str(trace.annotations['channel_index'])
#                    if np.isnan(trace).all() != np.nan:
#                        break
#                        v = trace
                    print (v)
                    plt.plot(v)
                    plt.title('recording type: '+str(trace.name).split("-")[0]+' '+'single_trace'+' '+'from the channel: '+ str(channel_index))
                    plt.ylabel('Amplitude of signal: '+str(trace[0]).split()[1])
                    plt.xlabel('time (mS)')
                    plt.tight_layout()
                    plt.legend()
                    plt.savefig(str(results_folder)+str(file_name)+"_"+str(trace.name)+"_"+"from_the_channel_"+channel_index+str(trace.t_start)+".png")
                    plt.show()
                    plt.close()
