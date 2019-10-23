#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 00:32:11 2019

@author: anzal
"""

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
                for signal in analog_signals:
                    for trace in signal.annotations.iteritems():
                        v= signal
                        print(v)
                        plt.plot(v)
                    plt.show()
                    plt.savefig(results_folder+file_name+" "+'_compiled.png')