#!/usr/bin/env python3
__author__           = "Anzal KS"
__copyright__        = "Copyright 2021-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"

from pathlib import Path
import neo.io as nio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import scipy.signal as signal
from itertools import islice 
from itertools import product
from pprint import pprint
import argparse
import collections
from matplotlib import gridspec

class Args: pass 
args_ = Args()

def list_files(p):
    p =Path(p)
    f_list = []
    f_list=list(p.glob('**/*txt'))
    return f_list


def load_text(txt):
    str(txt)
    print(txt)
    txt_read = np.loadtxt(txt, comments="#", delimiter=" ", dtype=int, unpack=True)
    text = txt_read
    fr_no = txt_read[0]
    row_size = txt_read[1]
    col_size = txt_read[2]
    br_sp_loc = txt_read[3:]
    fr_size = [row_size[0],col_size[0]]
    fr = np.zeros(fr_size, dtype=int)
    br_idx = []
    for i in br_sp_loc:
        indx = np.unravel_index(i-1,fr_size)
        br_idx.append(indx)
    print(br_idx)
    fr_1 = []
    for i in br_idx:
        br_ind = np.column_stack((i[0],i[1]))
        fr_1.append(br_ind)
        for bi,b in enumerate( br_ind):
            print(f"***{b}*****{b.dtype}***")
            fr[b]=1
            print(f"######{fr}#####")
#        fr[fr_1[i]]
#        print(fr)
#    print(f"~~~~~~~~~~`{fr_1}~~~~~~~~`")
#        ind_1 = []
#        for b in br_ind:
#            np.put(fr,br_ind,1)
#        np.put(fr,i,1)
#        print(f"@@@@@@@{fr}@@@@@@@@")
#            print(fr)

#make an array of size specified by array size for each frame number, put the
#coordinate value as one in each frame based on the columns, loof throuhg
#br_sp_loc for achieving it?
    return None

def main():
    # Argument parser.
    description = '''Script to plot the projection grids from polygon'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--folder-path', '-f'
            , required = False, default ='./', type=str
            , help = 'path of folder with grid info text'
            )
    args = parser.parse_args()
    fpaths = list_files( args.folder_path )
    for f in fpaths:
        load_text(f)





if __name__  == '__main__':
    main()

