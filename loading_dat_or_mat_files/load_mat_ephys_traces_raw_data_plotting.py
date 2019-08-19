## -*- coding: utf-8 -*-
#"""
#Spyder Editor
#
#This is a temporary script file.
#"""

#impost essential libraries
import os
import scipy.io as sio
import matplotlib.pyplot as plt

#set the directory path with files to read
folder_to_read = "/mnt/5D4B-FA71/Data/190814/"
os.mkdir(folder_to_read+'Results/')
results_folder = str(folder_to_read+'Results/')
#list out all the files with .mat extension for plotting
for root, dirs, files in os.walk(folder_to_read):
    for file in files:
        if file.endswith(".mat"):
            print(file)
            #import the file of interest
            file_to_read = sio.loadmat(root+file)
            for single_file in file_to_read:
                #list out all the keys in dictionary iported from mat file with "trace" labelled on it
                traces = list(filter (lambda x:'Trace' in x, file_to_read.keys()))
                #omit the usual keys which are not traces/time
                #loop through each trace/key to extract time and signal
                for trace in traces:
                    print (trace)
                    #time extrated
                    t=file_to_read[trace][:,0]*1000
                    #signal trace extracted
                    v=file_to_read[trace][:,1]*1000
                    #plots
                    plt.plot(t,v)
                    #plot's shown for each trace
                    plt.savefig(results_folder+file+trace+'.png')
                    plt.show()
for single_file in file_to_read:
    #list out all the keys in dictionary iported from mat file with "trace" labelled on it
    traces = list(filter (lambda x:'Trace' in x, file_to_read.keys()))
    #omit the usual keys which are not traces/time
    #loop through each trace/key to extract time and signal
    for trace in traces:
        print (trace)
        #time extrated
        t=file_to_read[trace][:,0]*1000
        #signal trace extracted
        v=file_to_read[trace][:,1]*1000
        #plots
        plt.plot(t,v)
        #plot's shown for each trace
    plt.savefig(results_folder+file+trace+'compiled_.png')
    plt.show()
