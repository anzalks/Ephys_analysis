#!/usr/bin/env python3
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
            m = [v,t]
            Vm_trail.append(m)
            total_time = float(tf-ti)
    return Vm_trail

def protol_trace(raw_protocol):
    pro_traces = raw_protocol[0]
    print(f"the length of raw prot trace = {len(pro_traces)}")
    for trace_no,trace in enumerate(pro_traces):
        print(f"###########......the {trace_no}th with length"
              f"{len(trace)}trace is {trace}.......############")

class file_details:
    def __init__(self,p):
        self.p= p
        self.day_folder = list_files(p)
        self.cell_list =  list(Path.iterdir(p))
        cells = self.cell_list
        experiments = []
        for cell,exp in enumerate(cells):
            experiment_list = [cell, list_files(exp)]
            experiments.append(experiment_list)
        self.experiments = experiments
        self.cell_data =[]
        self.cell_no = []
        for experiment in experiments:
            self.cell_no.append(experiment[0])
            self.cell_data.append(experiment[1])
#        pprint(f"cell no. =  {self.cell_no}, cell data = {self.cell_data}")



class protocols:
    def __init__(self,cell_no, cell):
        self.cell_data_files = cell
        cell_ = self.cell_data_files
        self.raw_trace = []
        self.raw_protocol = []
        for file_ in cell_:
            abf_file(file_)


def analysis1( protocol ):
    # do something
    print( "analysis1" )

def analysis2( protocol ):
    # do something
    print( "analysis2" )

def analysis3( protocol ):
    # do something
    print( "analysis3" )



def plot1( protocol ):
    # do something
    print( "plot1" )

def plot2( protocol ):
    # do something
    print( "plot2" )

def plot3( protocol ):
    # do something
    print( "plot3" )


proto_map = { "baseline": ( analysis1, plot1 ), "Rm": (analysis2, plot2), "pattern1": (analysis3, plot3 ) }

proto_sequence = [ "baseline", "Rm", "pattern1"]


class Protocol:
    def __init__(self, abf):
        self.abf = abf #Name of the file
        reader = nio.AxonIO(abf)
        self.name = abf.split("/")[-1]
        prot_name = reader._axon_info['sProtocolPath']
        prot_name = str(prot_name).split("\\")[-1]
#        prot_name = prot_name.split(".")[0]

        self.protocol_name = prot_name
        self.analysis_func = proto_map[prot_name][0]
        self.plot_func = proto_map[prot_name][1]

        self.recording = raw_trace(abf)
        self.protocol_raw_data = reader.read_raw_protocol()
        self.protol_trace = protol_trace(self.protocol_raw_data)
        self.sampling_rate = reader._sampling_rate
        self.protol_unit = self.protocol_raw_data[2][0]
        #self.supported_obj = reader.supported_objects
        print(f"++++++{self.abf_name}+++++++++")
        print(f".......{self.protocol_name}........")
        print(f"'''''''{self.sampling_rate}'''''''''''''")


def main():
    # Argument parser.
    description = '''Analysis script for abf files.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--folder-path', '-f'
            , required = False, default ='./', type=str
            , help = 'path of folder with  abf files '
            )
    args = parser.parse_args()
    fnames = os.listdir( args.folder_path )
    protocols = {}
    for f in fnames:
        if f[-4] == ".abf":
            print( f )
            proto = Protocol( f )
            protocols[proto.name] = proto

    # Do analysis
    for p in proto_sequence:
        protocols[p].analysis_func()

    # Do plotting
    for p in proto_sequence:
        protocols[p].plot_func()


if __name__  == '__main__':
    main()
