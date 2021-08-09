#!/usr/bin/env python3
__author__           = "Anal Kumar"
__copyright__        = "Copyright 2021-, Anal Kumar"
__maintainer__       = "Anal Kumar"
__email__            = "analkumar@ncbs.res.in"

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
    txt_read = np.loadtxt(txt, comments="#", delimiter=" ", dtype=int, unpack=True)
    txt_read = np.transpose(txt_read) #transposing so that the first index is the first frame

    Framedict = {}
    for i in range(len(txt_read)):
        fr_no = txt_read[i][0]
        num_of_Cols = txt_read[i][1]
        num_of_Rows = txt_read[i][2]
        brightidx = txt_read[i][3:]
        zeromatrix = np.zeros([num_of_Rows, num_of_Cols])
        brightidx2D = np.transpose(np.unravel_index(brightidx-1, (num_of_Rows,num_of_Cols))) #Getting 2d coordinates from linear coord
        for j in range(len(brightidx2D)):
            zeromatrix[brightidx2D[j][0]][brightidx2D[j][1]] = 1 #Replacing the zeors with 1s
        Framedict[fr_no] = zeromatrix

#    print(Framedict)
    return Framedict
def plot_projections(projections):
    for i in projections:
        for l in i:
            fr = i[l]
            plt.imshow(fr)
            plt.show(block=False)
#            plt.pause(0.05)
            plt.close()

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
    projections = []
    for f in fpaths:
        p = load_text(f)
        projections.append(p)
    plot_projections(projections)




if __name__  == '__main__':
    main()

