#!/usr/bin/env python3
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2019-, Dilawar Singh"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import neo.io

class Args : pass
args_ = Args()

def plot(bl):
    global args_
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,8))
    gridSize = (3, 2)
    ax1 = plt.subplot2grid( gridSize, (0,0), colspan = 2 )
    ax2 = plt.subplot2grid( gridSize, (1,0), colspan = 2 )
    ax3 = plt.subplot2grid( gridSize, (2,0), colspan = 1 )
    ax4 = plt.subplot2grid( gridSize, (2,1), colspan = 1 )

    ax1.set_title('Analog Singal')
    ax1.set_xlabel('Time (s)')

    ax2.set_title('Spike Train')

    for seg in bl.segments:
        print( f"[INFO ] SEG: {seg}" )
        for i, asig in enumerate(seg.analogsignals):
            times = asig.times.rescale('s').magnitude
            asig = asig.magnitude
            ax1.plot(times, asig)

        trains = [st.rescale('s').magnitude for st in seg.spiketrains]
        colors = plt.cm.jet(np.linspace(0, 1, len(seg.spiketrains)))
        ax2.eventplot(trains, colors=colors)

    plt.tight_layout()
    outfile = args_.output
    if outfile is None:
        outfile = args_.input + '.png'
    plt.savefig(f'{outfile}')

def main():
    global args_
    reader = neo.io.RawBinarySignalIO(args_.input
            , sampling_rate=args_.sampling_rate
            )
    bl = reader.read()[0]
    plot(bl)
        
if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''Analyze Hekka Dat file.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input', '-i'
            , required = True
            , help = 'Input file'
            )
    parser.add_argument('--output', '-o'
            , required = False
            , help = 'Output file'
            )
    parser.add_argument( '--threshold', '-t'
            , required = False , default = 0
            , type = float
            , help = 'Enable debug mode. Default 0, debug level'
            )
    parser.add_argument('--sampling-rate', '-R'
             , required = False, default = 20000 , type = int
             , help = 'Sampling rate.'
             )
    
    parser.parse_args(namespace=args_)
    main()

