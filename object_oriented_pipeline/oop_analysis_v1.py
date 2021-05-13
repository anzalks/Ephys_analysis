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
import os
import scipy.signal as signal
from itertools import islice 
from pprint import pprint
import argparse
import collections

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
    return Vm_trail

def protol_trace(raw_protocol):
    pro_traces = raw_protocol[0]
    print(f"the length of raw prot trace = {len(pro_traces)}")
    for trace_no,trace in enumerate(pro_traces):
        print("ptrace")

class File_details:
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







def find_baseline( recording ):
    r = recording[0]
    t = r[0]
    v = r[1]
    y_max = max(v)
    y_min = min(v)
    x_min = min(t)
    x_max = max(t)
    rmp = np.around(np.average(v), decimals = 2)
    plt.plot(t,v)
    plt.ylim(y_min-10, y_max+10 )
    plt.title(str(rmp))
    plt.show()
    print( "analysis1" )

def analysis2( protocol ):
    # do something
    print( "analysis2" )

def analysis3( protocol ):
    # do something
    print( "analysis3" )

def analysis4( protocol ):
    # do something
    print( "analysis1" )

def analysis5( protocol ):
    # do something
    print( "analysis2" )

def analysis6( protocol ):
    # do something
    print( "analysis3" )

def analysis7( protocol ):
    # do something
    print( "analysis1" )

def analysis8( protocol ):
    # do something
    print( "analysis2" )

def analysis9( protocol ):
    # do something
    print( "analysis3" )

def analysis10( protocol ):
    # do something
    print( "analysis1" )

def analysis11( protocol ):
    # do something
    print( "analysis2" )

def analysis12( protocol ):
    # do something
    print( "analysis3" )

def analysis13( protocol ):
    # do something
    print( "analysis2" )

def analysis14( protocol ):
    # do something
    print( "analysis3" )       


def Base_line_plot(f, fi, rec_trace, sampling_rate,trace_unit, protocol_unit):
    fig = plt.figure(figsize=(16,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    RMP = []
    iter_num = 0
    for v in  enumerate(rec_trace):
        trace_number = v[0]
        trace = v[1][0]
        time = v[1][1]
        mean_RMP = np.mean(trace)
        if mean_RMP <-50:
            iter_num +=1
            ax1.plot(time, trace, label=f'trace no. {iter_num}', alpha = 0.5)
            P_traces= protocols[0]
            RMP = mean_RMP
            for p in P_traces:
                for i in p:
                    t_ = len(i)/sampling_rate
                    t = np.linspace(0,float(t_), len(i))
                    ax2.plot(t,i)
#        First_inj = mpatches.Patch(color='green', label='First injection')
#        Thres_inj = mpatches.Patch(color='black', label='Threshold injection')
#        Last_inj = mpatches.Patch(color='red', label='Final injection')
#        ax2.legend(handles=[First_inj,Thres_inj,Last_inj])
    ax1.set_title('Recording')
    ax1.set_ylabel(trace_unit)
    ax1.set_xlabel('time(s)')
    ax1.legend()
    ax1.set_ylim(-90,-20)
    ax2.set_title('Protocol trace')
    ax2.set_ylabel(protocol_unit)
    ax2.set_xlabel('time(s)')
    #        ax2.legend()
    plt.figtext(0.10,-0.05, "Resting membrane potential average from"
                f" {iter_num} traces= "+
                str(np.around(RMP,decimals = 2))+" mV", fontsize=12, va="top", ha="left")
    plt.suptitle(f'Protocol type: {protocol_type} - {protocol_used}',fontsize=15)
    plt.figtext(0.10, -0.10, f"sampling rate = {sampling_rate}" ,
                fontsize=12, va="top", ha="left" )
    plt.figtext(0.10, -0.15, f"total recording time = {total_time} s" ,
                fontsize=12, va="top", ha="left")
    outfile = str(outdir)+"/"+str(f.stem)+ f" {protocol_used}_{fi}.png"
    plt.savefig(outfile,bbox_inches = 'tight')
    print("-----> Saved to %s" % outfile)
    fig = plt.close()


def plot2( protocol ):
    # do something
    print( "analysis2" )

def plot3( protocol ):
    # do something
    print( "analysis3" )

def plot4( protocol ):
    # do something
    print( "analysis1" )

def plot5( protocol ):
    # do something
    print( "analysis2" )

def plot6( protocol ):
    # do something
    print( "analysis3" )

def plot7( protocol ):
    # do something
    print( "analysis1" )

def plot8( protocol ):
    # do something
    print( "analysis2" )

def plot9( protocol ):
    # do something
    print( "analysis3" )

def plot10( protocol ):
    # do something
    print( "analysis1" )

def plot11( protocol ):
    # do something
    print( "analysis2" )

def plot12( protocol ):
    # do something 
    print( "analysis3" ) 

def plot13( protocol ):
    # do something 
    print( "analysis3" ) 

def plot14( protocol ):
    # do something 
    print( "analysis3" ) 


proto_map = { "1_RMP_gap_free_1min": ( find_baseline, Base_line_plot ), 
             "2_Input_resistance_50pA_300ms": (analysis2, plot2), 
             "3_cell_threshold_10pA_step_-50_140pA_500ms": (analysis3, plot3 ),
             "4_Baseline_point_by_point_response_25_x_25_grid"
             "_size_50ms_pulse_300ms_spaced":(analysis4,plot4),
             "5_Baseline_Excitation_point_by_point_response_25"
             "_x_25_grid_size_50ms_pulse_200ms_spaced":(analysis5,plot5),
             "6_Baseline_Inhibition_point_by_point_response"
             "_25_x_25_grid_size_50ms_pulse_200ms_spaced":(analysis6,plot6),
             "7_10_patterns_Base_line_2s_50ms_frames_3"
             "_times_repeat":(analysis7,plot7),
             "8_10_Pattern_baseline_excitation_0mV_holding"
             "_25X25_single_point":(analysis8,plot8),
             "9_10_Pattern_baseline_Inhibition-70mV_holding"
             "_25X25_single_point":(analysis9,plot9),
             "10_Pattern_training_every_2s_50ms_frames_3_times"
             "_repeat_one_depolarised_frame":(analysis10,plot10),
             "11_Inhibition_point_by_point_response_25_x_25"
             "_grid_size_50ms_pulse_200ms_spaced":(analysis11,plot11),
             "12_Inhibition_point_by_point_response_25_x_25"
             "_grid_size_50ms_pulse_200ms_spaced":(analysis12,plot12),
             "13_10_Pattern_Inhibition-70mV_holding_25X25_single_point":(analysis13,plot13),
             "14_10_Pattern_Inhibition-70mV_holding_25X25_single_point":(analysis14,plot14)

            }

proto_sequence = ["1_RMP_gap_free_1min", 
                  "2_Input_resistance_50pA_300ms", 
                  "3_cell_threshold_10pA_step_-50_140pA_500ms",
                  "4_Baseline_point_by_point_response_25_x_25_grid"
                  "_size_50ms_pulse_300ms_spaced",
                  "5_Baseline_Excitation_point_by_point_response_25"
                  "_x_25_grid_size_50ms_pulse_200ms_spaced",
                  "6_Baseline_Inhibition_point_by_point_response"
                  "_25_x_25_grid_size_50ms_pulse_200ms_spaced",
                  "7_10_patterns_Base_line_2s_50ms_frames_3"
                  "_times_repeat",
                  "8_10_Pattern_baseline_excitation_0mV_holding"
                  "_25X25_single_point",
                  "9_10_Pattern_baseline_Inhibition-70mV_holding"
                  "_25X25_single_point",
                  "10_Pattern_training_every_2s_50ms_frames_3_times"
                  "_repeat_one_depolarised_frame",
                  "11_Inhibition_point_by_point_response_25_x_25"
                  "_grid_size_50ms_pulse_200ms_spaced",
                  "12_Inhibition_point_by_point_response_25_x_25"
                  "_grid_size_50ms_pulse_200ms_spaced",
                  "13_10_Pattern_Inhibition-70mV_holding_25X25_single_point",
                  "14_10_Pattern_Inhibition-70mV_holding_25X25_single_point"
                 ] 


class Protocol:
    def __init__(self, abf):
        self.abf = abf #Name of the file
        reader = nio.AxonIO(abf)
        self.name = abf.split("/")[-1]
        prot_name = reader._axon_info['sProtocolPath']
        prot_name = str(prot_name).split("\\")[-1]
        prot_name = prot_name.split(".")[-2]

        self.protocol_name = prot_name
        self.analysis_func = proto_map[prot_name][0]
        self.plot_func = proto_map[prot_name][1]
        self.recording = raw_trace(abf)
        self.protocol_raw_data = reader.read_raw_protocol()
        self.protol_trace = protol_trace(self.protocol_raw_data)
        self.sampling_rate = reader._sampling_rate
        self.protol_unit = self.protocol_raw_data[2][0]
        #self.supported_obj = reader.supported_objects


def main():
    # Argument parser.
    description = '''Analysis script for abf files.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--folder-path', '-f'
            , required = False, default ='./', type=str
            , help = 'path of folder with  abf files '
            )
    args = parser.parse_args()
    fpaths = list_files( args.folder_path )
#    print(fpaths)
    protocols = {}
    for f in fpaths:
        f =str(f)
        if f.split(".")[-1] == "abf":
            try:
                proto = Protocol( f )
                protocols[proto.protocol_name] = proto
            except:
                continue

    # Do analysis
    for p in proto_sequence[0:2]:
        protocols[p].analysis_func(protocols[p].recording)
    # Do plotting
#    for p in proto_sequence:
#        protocols[p].plot_func()

if __name__  == '__main__':
    main()
