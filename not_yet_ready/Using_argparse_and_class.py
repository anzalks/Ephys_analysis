#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:47:02 2019

@author: anzal
"""

import argparse
import os
import neo.io as nio
import numpy as np
import matplotlib.pyplot as plt
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor
from scipy.signal import savgol_filter as filter_s


parser = argparse.ArgumentParser(description='Program that plots analysis performed on .dat files for ephys data')
parser.add_argument('file_path', help = 'String specifying the folder path with all the data in it')
args = parser.parse_args()


    def file_lister(folder_to_read):
        file_list = []
        for root, dirs, files in os.walk(folder_to_read):
            for file in files:
                if file.endswith(".dat"):
                    file_list.append(os.path.join(root,file))
                
#                print (file_list)
                file_list.append(root+file)
                print (file_list)