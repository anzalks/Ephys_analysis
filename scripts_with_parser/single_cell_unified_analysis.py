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
        protocol_name = reader._axon_info['sProtocolPath']
        protocol_name = str(protocol_name).split("\\")[-1]
        protocol_name = protocol_name.split(".")[0]
        protocol_num = protocol_name.split("_")[0]
        if str.isdigit(protocol_num) == True:
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
    print(f"number of protocols used in total = {protocol_num}")
    print(f"names of protocols used in total = {protocol_name}")
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

def raw_trace_plot(f, folder_path):
# use folder path in the previous loop to make use of the cell_*** folder path
# allocation 
#    columns = int(len(Vm_trail)/3)
    f = str(f)
    Vm_trail = []
    reader = nio.AxonIO(filename=f)
    segments = reader.read_block().segments
    sample_trace = segments.
     rows = 3
 #    v = np.reshape(Vm_trail[1],(columns,rows))
 #    t = np.reshape(Vm_trail[0],(columns,rows))
 #    print(v.shape())
     print("_____________________")
 #    print(f"reminder = {columns}")
     fig, ax_array = plt.subplots(rows,2)
     va_,vb_,vc_ = [],[],[]
     ta_,tb_,tc_ = [],[],[]
     na,nb,nc = 0,0,0
     print(f"na {na} nb {nb} nc {nc}")

     for ind_i,i in enumerate(Vm_trail):
         r = ind_i%rows
         a,b,c =(1%3,2%3,0)
         v = Vm_trail[ind_i][0]
         t = Vm_trail[ind_i][1]
         if r == a:
             na +=1
             ta_.append(t)
             va_.append(v)
             ax_array[2][0].plot(t,v, alpha = 0.5)
             if na == 3:
                 va_ = np.mean(va_, axis = 0)
                 ta_ = np.mean(ta_, axis = 0) 
 #                print(ta_,va_)
                 ax_array[2][0].plot(ta_,va_, color = 'b', linewidth=0.5)
 
 
 #            ax_array[2].plot(Vm_trail[ind_i][1],Vm_trail[ind_i][0])
 #            plt.subplot(rows,columns,ind_i)
 #            plt.plot(Vm_trail[ind_i][1],Vm_trail[ind_i][0])
         elif r == b:
             nb +=1
             vb_.append(v)
             tb_.append(t)
             ax_array[1][0].plot(t,v, alpha = 0.5)
             if nb == 3:
                 vb_ = np.mean(vb_, axis = 0)
                 tb_ = np.mean(tb_, axis = 0) 
 #                print(tb_,vb_)
                 ax_array[1][0].plot(tb_,vb_, color = 'k', linewidth=0.5)
 #                ax_array[1].plot(Vm_trail[ind_i][1],Vm_trail[ind_i][0])
 #            plt.subplot(rows,columns,ind_i+1)
 #            plt.plot(Vm_trail[ind_i][1],Vm_trail[ind_i][0])

        elif r == b:
             nb +=1
             vb_.append(v)
             tb_.append(t)
             ax_array[1][0].plot(t,v, alpha = 0.5)
             if nb == 3:
                 vb_ = np.mean(vb_, axis = 0)
                 tb_ = np.mean(tb_, axis = 0) 
 #                print(tb_,vb_)
                 ax_array[1][0].plot(tb_,vb_, color = 'k', linewidth=0.5)
 #                ax_array[1].plot(Vm_trail[ind_i][1],Vm_trail[ind_i][0])
 #            plt.subplot(rows,columns,ind_i+1)
 #            plt.plot(Vm_trail[ind_i][1],Vm_trail[ind_i][0])
         elif r == c:
             nc +=1
             vc_.append(v)
             tc_.append(t)
             ax_array[0][0].plot(t,v, alpha = 0.5, label=f'trial {nc}')
             v_lowercut = np.copy(v)
 #            v_lowercut[v_lowercut<-50] = -50
             time = np.copy(t)
             v_smooth = v_lowercut
 #            v_smooth = np.ma.average(v_lowercut,axis=0)
 #            v_smooth = butter_bandpass_filter(v_lowercut,1,20, sampling_rate, order=1)
             peaks = signal.find_peaks(x=v_smooth, height=-80,  threshold=None,
                                       distance=50,
                                       prominence=2.5, width=100, wlen=None,
                                       rel_height=1,plateau_size=None)
 #            peaks_v = v_smooth
 #            peaks_t = time
             peaks_t = t[peaks[0]-10]
             peaks_v = v[peaks[0]-10]
             ax_array[0][1].scatter(peaks_t,peaks_v,alpha=0.5, marker='.', 
                                    label = f'peak response in trial {nc}')
             print(f"peaks = {peaks_v}, {peaks_t}")
             if nc == 3:
                 vc_ = np.mean(vc_, axis = 0)
                 tc_ = np.mean(tc_, axis = 0) 
 #                print(tc_,vc_)
                 ax_array[0][0].plot(tc_,vc_, color = 'r', linewidth=0.5, label
                                    = f'mean response for {nc} trails')
                 ax_array[0][1].plot(tc_,vc_, color = 'r',alpha=0.3, linewidth=0.2)
 #            plt.subplot(rows,columns,ind_i+2)
 #            plt.plot(Vm_trail[ind_i][1],Vm_trail[ind_i][0])
 
     ax_array[2][0].set_title('TTL')
     ax_array[2][0].set_xlabel('time(s)')
 #    ax_array[2].set_ylim(-0.5,2.5)
 #    ax_array[2].set_xlim(0.2,0.5)
     ax_array[2][0].set_ylabel('V')
     ax_array[1][0].set_title('photo diode')
 #    ax_array[1].set_ylim(2,7.5)
 #    ax_array[1].set_xlim(0.2,0.5)
     ax_array[1][0].set_ylabel('pA')
     ax_array[0][0].set_title('cell trace')
     ax_array[0][0].set_ylim(-100,5)
     ax_array[0][0].set_ylabel('mV')
 #    ax_array[0].set_xlim(0.2,0.5)
 #    plt.title("three channel recordings")
     ax_array[0][1].set_title('Response peak spread')
     ax_array[0][1].set_xlabel('time(s)')
     ax_array[0][1].set_ylim(-100,5)
 
     ax_array[1,1].axis('off')
     ax_array[2,1].axis('off')
     fig.legend(title="Legend",borderaxespad=1, loc="lower right")
     plt.tight_layout()
 
     outfile  savefig(*args, **kwargs) em)+ f" blipo_col_realing_1_resized.png"
     plt.savefig(outfile,bbox_inches = 'tight',dpi = (300))
     plt.close()

























def total_trace(f):
    global Vm_trail
    global sampling_rate
    global trace_unit
    global toal_time
    f= str(f)
    Vm_trail = []
    reader = nio.AxonIO(filename=f)
    segments = reader.read_block().segments
    sample_trace = segments[0].analogsignals[0]
    sampling_rate = sample_trace.sampling_rate
    trace_unit = str(sample_trace.units).split()[1]
#    print(sampling_rate)
#    print(trace_unit)
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
    del(reader)
    return Vm_trail

def series_res_check(f):
    fig = plt.figure(figsize=(16,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)




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
    for cell in cell_set:
        print(f"protocol set length  --> {len(cell)}")
        for pn, protocol in enumerate (cell):
#plotting fucntion for each of the protocols            
            print(f"path to the files = no.{pn} , {protocol}")

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


