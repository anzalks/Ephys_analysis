#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 23:31:28 2019

@author: anzal
"""
import argparse
description = ''''Analysis script for current clamp experiments. Had to be run from the command line'''
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--file-path', '-f',required=True, type=str, help=' Input the file path of the folder containing .dat files for analysis ')
#set the directory path with files to read
args = vars(parser.parse_args())

print (args['file_path'])





