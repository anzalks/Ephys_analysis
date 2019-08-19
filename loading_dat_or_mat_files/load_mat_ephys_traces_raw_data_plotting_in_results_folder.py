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
folder_to_read = "/mnt/5D4B-FA71/Data/190801/"
#make a folder to save all the results
os.mkdir(folder_to_read+'Results/')
#make the file path
results_folder = str(folder_to_read+'Results/')
#list out all the files with .mat extension for plotting
for root, dirs, files in os.walk(folder_to_read):
    for file in files:
        if file.endswith(".mat"):
            print(file)
            #load the file of interest
            file_to_read = sio.loadmat(root+file)
            print(file_to_read)
            #make a list of all files to iterate through
            file_to_read_list = list(file_to_read)
            print (file_to_read_list)
            #list out all the keys with "trace" in it
            traces_in_file_to_read = list(filter (lambda x:'Trace' in x, file_to_read.keys()))
            print (traces_in_file_to_read)
            #iterate through each trace in each file 
            for each_set_of_traces in traces_in_file_to_read:
                print(each_set_of_traces)
                #convert the loop index trace to string so that we can access it as its key
                each_set_of_traces = str(each_set_of_traces)
                print (each_set_of_traces)
                #access the array using the key of interest and access data showing time in it
                t = file_to_read[each_set_of_traces][:,0]*1000
                print(t)
                #access the array and extract the analogue signal voltage/current
                v = file_to_read[each_set_of_traces][:,1]*1000
                print(v)
                #plot the graph
                plt.plot(t,v)
            #save the figure in the results folder with png as extension, fiile name, Key detail(trace number)     
            plt.savefig(results_folder+file+each_set_of_traces+'_compiled.png')
            plt.show() 
            for each_set_of_traces in traces_in_file_to_read:
                print(each_set_of_traces)
                each_set_of_traces = str(each_set_of_traces)
                print (each_set_of_traces)
                t = file_to_read[each_set_of_traces][:,0]*1000
                print(t)
                v = file_to_read[each_set_of_traces][:,1]*1000
                print(v)
                plt.plot(t,v)
                plt.savefig(results_folder+file+each_set_of_traces+'.png')
                plt.show()