#!/usr/bin/env python3
__author__           = "Anzal KS"
__copyright__        = "Copyright 2019-, Anzal KS"
__maintainer__       = "Anzal KS"
__email__            = "anzalks@ncbs.res.in"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from pathlib import Path
import neo.io as nio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import neo.io as nio
import scipy.signal as signal
from itertools import islice 

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

#def result_dir(p):
#    global outdir
#    outdir = p/'results'
#    outdir.mkdir(exist_ok=True, parents=True)

def data_file_filter(f_list):
    r = []
#    print(f"length of file list beofore{len(f_list)}")
    for f in f_list:
        p=str(f)
        reader = nio.AxonIO(p)
        prot_name = reader._axon_info['sProtocolPath']
        prot_name = str(prot_name).split("\\")[-1]
        prot_name = prot_name.split(".")[0]
        prot_num = prot_name.split("_")[0]
        if str.isdigit(prot_num) == True:
            r.append(f)
    f_list = r
    f_list.sort()
    del(reader)
#    print(f"length of file list later{len(f_list)}")
    return f_list



def protocol_name(f):
    f=str(f)
    reader = nio.AxonIO(f)
    protocol_name = reader._axon_info['sProtocolPath']
    protocol_name = str(protocol_name).split("\\")[-1]
    protocol_name = protocol_name.split(".")[0]
    protocol_num = protocol_name.split("_")[0]
    del(reader)
    return protocol_num, protocol_name

def tot_prot_num(f_list):
    len_prot = []
    for f in f_list:
        f= str(f)
        m = protocol_name(f)[0]
        len_prot.append(m)
    len_prot
    total_protocols=list(set(len_prot))
    total_protocols.sort()
    return total_protocols

def set_of_exp(f_list,expt_num):
    expt_num = int(len(expt_num))
    i=0
    cell_set = []
    while i<len(f_list):
        x=f_list[i:i+expt_num]
        i = i+expt_num
        cell_set.append(x)
    return cell_set

def cell_sorted_results(outdir,cell_set):
    cell_nos = list(np.arange(len(cell_set)))
    for cell in cell_nos:
        Path.mkdir(Path.joinpath(outdir,f'cell_{cell}'),exist_ok=True, parents=True)

def plot_selector(protocol_index):
    protocol_index = int(protocol_index)
    similar_plots = [7,]
    if protocol_index ==1:
        plot_function = "series_res_check"
    elif protocol_index ==2:
        plot_function = "rmp"
    elif protocol_index ==3:
        plot_function  = "neuron_threshold"
    elif protocol_index ==4:
        #same function can be used for before and after training protocols use
        #any in thatcase
        plot_function = "baseline_points"
    elif protocol_index ==5:
        #same function can be used for before and after training protocols use
        #any in thatcase
        plot_function = "baseline_pattern"
    elif protocol_index ==6:
        #same function can be used for before and after training protocols use
        #any in thatcase
        plot_function = "e_point"
    elif protocol_index ==7:
        #same function can be used for before and after training protocols use
        #any in thatcase
        plot_function = "e_pattern"
    elif protocol_index ==8:
        #same function can be used for before and after training protocols use
        #any in thatcase
        plot_function = "i_point"
    elif protocol_index ==9:
        #same function can be used for before and after training protocols use
        #any in thatcase
        plot_function = "i_pattern"
    elif protocol_index ==10:
        #same function can be used for before and after training protocols use
        #any in thatcase
        plot_function = "training"
    else:
        plot_function = "something not a protocol"
    return plot_function



def raw_trace_plot(f):
# use folder path in the previous loop to make use of the cell_*** folder path
# allocation 
#    columns = int(len(Vm_trail)/3)
    f = str(f)
    Vm_trail = []
    reader = nio.AxonIO(filename=f)
    segments = reader.read_block().segments
    sample_trace = segments[0].analogsignals[0]
    sampling_rate = sample_trace.sampling_rate
    trace_unit = str(sample_trace.units).split()[1]
    print(f"length of segments = {len(segments)}")
    print(sampling_rate)
    print(trace_unit)
    for si, segment in enumerate(segments):
        analog_signals = segment.analogsignals
        for trace in analog_signals:
            print(f"length of each trace in the segments = {len(trace) }")
            v = trace
            v = np.ravel(v)
            v = v.magnitude
            tf = trace.t_stop
            ti = trace.t_start
            t = np.linspace(0,float(tf - ti), len(v))
            m = [v,t]
            Vm_trail.append(m)
            total_time = float(tf-ti)









def main(**kwargs):
    p = Path(kwargs['folder_path'])
    outdir = p/'results'
    outdir.mkdir(exist_ok=True,parents=True)
#    f_list = data_file_filter(p)
    f_list = list_files(p)
    f_list = data_file_filter(f_list)
    expt_num = tot_prot_num(f_list)
    cell_set = set_of_exp(f_list,expt_num)
    cell_sorted_results(outdir,cell_set)
#    cell set will give the set of files which are of different protocols but
#    from the same experiment.
    for cell in cell_set:
        print(f"protocol set length  --> {len(cell)}")
# f will have files in one set of protocols from a single experiment.        
        for f in cell:
            print(str(f))
            f_prot = protocol_name(f)
            print(f"protocol name = {f_prot[1]} protocol index number ="
                  f"{f_prot[0]}")
            protocol_index = f_prot[0]
            print(f"protocol index = {protocol_index}")
            prot = plot_selector(protocol_index)
            print(f"selected protocol to plot = {prot}")
            raw_trace_plot(f)
#        for pn, protocol in enumerate (cell):
#
#plotting fucntion for each of the protocols            
#            print(f"path to the files = no.{pn} , {protocol}")
#            for f in cell_set:
#                f = str(f)
#                raw_trace_plot(f)
#
#    print(f_list)
#    expt_num = tot_prot_num(f_list)
#    expt_num = expt_num
#    cell_set = set_of_exp(f_list,expt_num)
#    print(expt_num)
#    print(len(cell_set))
#    for fi, f in enumerate(f_list):
#        print(f"analysing file: {str(f)} ")
#        print(protocol_name(f))
#
#        total_trace(f)
#        multi_chan_plot(Vm_trail,f,sampling_rate)
#        protocol_class(f)
#        protocol_ckecker(protocol_type, protocols)
#
#        prot_num = Channel_fetcher(f)
#        print(f'******{seg_no}')
#        print(prot_num)
#        if prot_num >= 3:
#            multi_can_plot(Vm_trail,f)
#            iu = total_trace(f)
#            pprint(iu)

#            total_trace(f)
#            print("Multi channel signal")
#            f = str(f)
#            reader = nio.AxonIO(f)
#            list_d = reader._axon_info['listADCInfo']
#            print(len(reader.read_block(signal_group_mode='split-all').segments[0].analogsignals))
#            print("#######^^^^^^")
##            print(reader.channel_name_to_index(['IN0', 'FrameTTL', 'I_MTest2']))
#            for i in list_d:
#                chan_info = reader.header['signal_channels'][0]
##                print(chan_info)
#                trace_structure = total_trace(f)
##                print(trace_structure)
#                chan_name = str(i['ADCChNames']).split("'")[1]
#                



#        if protocol_used == 'Threshold_check':
#            continue
#            threshold_protocol(f, fi, Vm_trail, sampling_rate,trace_unit,
#                               protocol_unit)
#        elif protocol_used == 'Base_line_V':
#            Base_line_protocol(f, fi, Vm_trail, sampling_rate,trace_unit, protocol_unit)
#        elif protocol_used == 'input_res_check':
#            input_R_protocol(f, fi, Vm_trail, sampling_rate,trace_unit, protocol_unit)
#        else:
#            print("not threshold")

if __name__  == '__main__':
    import argparse
    # Argument parser.
    description = '''Analysis script for abf files.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--folder-path', '-f'
            , required = False, default ='./', type=str
            , help = 'path of folder with  abf files '
            )
    parser.parse_args(namespace=args_)
    main(**vars(args_)) 


