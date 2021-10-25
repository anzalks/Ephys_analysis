__author__           = "Anzal KS"
__copyright__        = "Copyright 2019-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"

from pathlib import Path
import neo.io as nio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import neo.io as nio
import scipy.signal as signal
from itertools import islice 
from pprint import pprint
import argparse

class Args: pass
args_ = Args()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(N=order, Wn=[low, high], btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def list_files(p):
    p =Path(p)
    f_list = []
    f_list=list(p.glob('**/*abf'))
    return f_list

def raw_trace(f):
    Vm_trail = []
    reader = nio.AxonIO(filename=f)
    segments = reader.read_block().segments
    sample_trace = segments[0].analogsignals[0]
    sampling_rate = sample_trace.sampling_rate
    trace_unit = str(sample_trace.units).split()[1]
    for si, segment in enumerate(segments):
        analog_signals = segment.analogsignals
        for trace in analog_signals:
            v = trace
            v = np.ravel(v)
            v = v.magnitude
            tf = trace.t_stop
            ti = trace.t_start
            t = np.linspace(0,float(tf - ti), len(v))
            m = [t,v]
            Vm_trail.append(m)
            total_time = float(tf-ti)
    print(f"diemntion of vm trail = {len(Vm_trail)}")
    return Vm_trail



def Rmp( recording ):
    rmp_result = []
    print(f"recording len ={len(recording)}")
    for i, r in enumerate(recording):
        print(i,r)
        t = r[0]
        v = r[1]
        y_max = max(v)
        y_min = min(v)
        x_min = min(t)
        x_max = max(t)
        rmp = np.round(np.average(v), decimals = 2)
        std_rmp = np.round( np.std(v), decimals = 2)
        print( f"resting membrane potential = {str(rmp)} std = {str(std_rmp)}" )
        d = rmp,std_rmp,"rmp_data"
        rmp_result.append(d)
    return rmp_result

def Rmp_plot(recording):
    fig = plt.figure(figsize=(8,5))
    for r in enumerate(recording):
        v = r[1]
        t = r[0]
        plt.plot(t,v)
    plt.savefig()












def sampling_rate(f):
    reader = nio.AxonIO(filename=f)
    segments = reader.read_block().segments
    sampling_rate = sample_trace.sampling_rate
    return sampling_rate


def main():
    # Argument parser.
    description = '''Analysis script for abf files.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--folder_files', '-f'
            , required = False, default ='./', type=str
            , help = 'path of folder with  abf files '
            )
    args = parser.parse_args()
    fpaths = list_files( args.folder_files )
    
    for f in fpaths:
        f = str(f)
        rmp = Rmp(f)
        print(f"resting membrane potential  = {rmp}")






if __name__  == '__main__':
    main()
