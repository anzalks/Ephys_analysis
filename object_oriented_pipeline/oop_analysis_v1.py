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
    print(f"diemntion of vm trail = {len(Vm_trail)}")
    return Vm_trail

def protol_trace(raw_protocol):
    pro_traces = raw_protocol[0]
    print(f"the length of raw prot trace = {len(pro_traces)}")
#    for trace_no,trace in enumerate(pro_traces):
#        print("ptrace")

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


def Rmp( recording,sampling_rate ):
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

def Series_res( recording, sampling_rate ):
    print(f"length of recording in series res check {len(recording)}")
    mean_R = []
    sampling_rate = sampling_rate
    for i, r in enumerate(recording):
        r = recording[i]
        t = r[0]
        v = r[1]
        trace = v
        time = t
        Vl= np.mean(trace[int(sampling_rate*0.5):int(sampling_rate*0.55)])
        Vb= np.mean(trace[int(sampling_rate*0.15):int(sampling_rate*0.20)])
        print (f"reference points = {Vl, Vb}")
        input_R = (((Vl-Vb)*1000)/(-50))
        print(f"input_resistance = {input_R}")
        mean_R.append(input_R)
    mean_R =np.round(np.mean(mean_R), decimals=2)
    input_R = (mean_R - 3.5)
    print (f" series_res = {mean_R} input res = {input_R}" )



def Threshold_check( recording,sampling_rate):
    print(f"length of recording in threshold check {len(recording)}")
    thresh_state =0
    sampling_rate = sampling_rate
    for i, r in enumerate(recording):
        r = recording[i]
        t = r[0]
        v = r[1]
        trace = v
        time = t
        if thresh_state ==0:
            v_smooth = butter_bandpass_filter(trace,1,500,sampling_rate, order =1)
            peaks, peak_dict = signal.find_peaks(x=v_smooth, height=None,
                                                threshold=None,
                                                distance=None, prominence=5,
                                                width=None, wlen=None,
                                                 rel_height=0.5,
                                                plateau_size=None)
            v_peaks = v_smooth
            t_peaks =t
            thr_peak = 3
            if len(peaks)!=0:
                thresh_state =1
                dv = np.diff(v_smooth)
                dt = np.diff(t)
                dv_dt = dv/dt
                dv_dt_max = np.max(dv_dt)
                v_dt_max = np.where(dv_dt == dv_dt_max)[0]
                t_dt_max = np.where(dv_dt == dv_dt_max)[0]
                print(f"max dv_dt locci = {v_dt_max , t_dt_max}")

def Base_line_grids(recording,sampling_rate):
    print(recording)
    rows = 3
    va_,vb_,vc_ = [],[],[]
    ta_,tb_,tc_ = [],[],[]
    na,nb,nc = 0,0,0
    for ind_i,i in enumerate(recording):
        r = ind_i%rows
        a,b,c =(1%3,2%3,0)
        v = recording[ind_i][1]
        t = recording[ind_i][0]
        if r == a:
            na +=1
            ta_.append(t)
            va_.append(v)
            if na == 3:
                va_ = np.mean(va_, axis = 0)
                ta_ = np.mean(ta_, axis = 0) 
#                print(ta_,va_)
            elif r == b:
                nb +=1
                vb_.append(v)
                tb_.append(t)
                if nb == 3:
                    vb_ = np.mean(vb_, axis = 0)
                    tb_ = np.mean(tb_, axis = 0) 
                    #                print(tb_,vb_)
            elif r == c:
                nc +=1
                vc_.append(v)
                tc_.append(t)
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
                print(f"peaks = {peaks_v}, {peaks_t}")
                if nc == 3:
                    vc_ = np.mean(vc_, axis = 0)
                    tc_ = np.mean(tc_, axis = 0)
                    #                print(tc_,vc_)

    print(f"length of rec 1 {recording[0]}")
    print(f"length of recording in {len(recording)}")


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


def Rmp_plot(recording,sampling_rate):
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
     plt.title("Resting membrane voltage")
     plt.xlabel("time (s)")
     plt.ylabel("membrane voltage (mV)")
     plt.figtext(0,0, f"resting membrane potential = {rmp} mV")
     plt.show()
     print( "analysis1" )

def Series_res_plot( recording, sampling_rate ):
    print(f"length of recording in series res check {len(recording)}")
    mean_R = []
    sampling_rate = sampling_rate
    for i, r in enumerate(recording):
        r = recording[i]
        t = r[0]
        v = r[1]
        y_max = max(v)
        y_min = min(v)
        x_min = min(t)
        x_max = max(t)
        trace = v
        time = t
        plt.plot(t,v)
        Vl= np.mean(trace[int(sampling_rate*0.5):int(sampling_rate*0.55)])
        Vb= np.mean(trace[int(sampling_rate*0.15):int(sampling_rate*0.20)])
        print (f"reference points = {Vl, Vb}")
        input_R = (((Vl-Vb)*1000)/(-50))
        print(f"input_resistance = {input_R} MOhm")
        mean_R.append(input_R)
    mean_R =np.round(np.mean(mean_R), decimals=2)
    input_R = (mean_R - 3.5)
    plt.ylim = (y_min-10, y_max+10)
    plt.title("Series resistance")
    plt.figtext(0,0, f"input res = {input_R} & series re = {mean_R} ",fontsize= 10)
    plt.xlabel("time (s)")
    plt.ylabel("membrane voltage (mV)")
    plt.show()
    print (f" series_res = {mean_R} input res = {input_R}" )

def Threshold_check_plot( recording,sampling_rate):
    print(f"length of recording in threshold check {len(recording)}")
    thresh_state =0
    sampling_rate = sampling_rate
    for i, r in enumerate(recording):
        r = recording[i]
        t = r[0]
        v = r[1]
        y_max = max(v)
        y_min = min(v)
        x_min = min(t)
        x_max = max(t)
        trace = v
        time = t
        if thresh_state ==0:
            v_smooth = butter_bandpass_filter(trace,1,500,sampling_rate, order =1)
            peaks, peak_dict = signal.find_peaks(x=v_smooth, height=None,
                                                threshold=None,
                                                distance=None, prominence=5,
                                                width=None, wlen=None,
                                                 rel_height=0.5,
                                                plateau_size=None)
            v_peaks = v_smooth
            t_peaks =t
            if len(peaks)!=0:
                thresh_state =1
                dv = np.diff(v_smooth)
                dt = np.diff(t)
                dv_dt = dv/dt
                dv_dt_max = np.max(dv_dt)
                v_dt_max = np.where(dv_dt == dv_dt_max)[0]-20
                t_dt_max = np.where(dv_dt == dv_dt_max)[0]-20
                plt.plot(t,v)
                plt.scatter(time[peaks[0]-10],trace[peaks[0]-10], color='r')
                plt.scatter(time[t_dt_max],trace[v_dt_max], color='k')
                print(f"max dv_dt locci = {v_dt_max , t_dt_max}")
    plt.ylim = (y_min-10, y_max+10)
    plt.title("Cell threshold")
    plt.figtext(0, 0, f"threshold voltage =  mV ",fontsize= 10)
    plt.xlabel("time (s)")
    plt.ylabel("membrane voltage (mV)")
    plt.show()

def Base_line_grids_plot(recording,sampling_rate):
    print(recording)
    rows = 3
    va_,vb_,vc_ = [],[],[]
    ta_,tb_,tc_ = [],[],[]
    na,nb,nc = 0,0,0
    fig, ax_array = plt.subplots(rows,2)
    for ind_i,i in enumerate(recording):
        r = ind_i%rows
        a,b,c =(1%3,2%3,0)
        v = recording[ind_i][1]
        t = recording[ind_i][0]
        if r == a:
            na +=1
            ta_.append(t)
            va_.append(v)
            ax_array[0][0].plot(t,v, alpha = 0.5)
            if na == 3:
                va_ = np.mean(va_, axis = 0)
                ta_ = np.mean(ta_, axis = 0) 
#                print(ta_,va_)
                ax_array[0][0].plot(ta_,va_, color = 'b', linewidth=0.5)
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
        elif r == c:
            nc +=1
            vc_.append(v)
            tc_.append(t)
            ax_array[2][0].plot(t,v, alpha = 0.5, label=f'trial {nc}')
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
                ax_array[0][1].plot(tc_,vc_, color = 'r', linewidth=0.5, label
                                    = f'mean response for {nc} trails')
                ax_array[0][1].plot(tc_,vc_, color = 'r',alpha=0.3, linewidth=0.2)
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
    plt.show()


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


proto_map = { "1_RMP_gap_free_1min": ( Rmp, Rmp_plot ), 
             "2_Input_resistance_50pA_600ms": (Series_res, Series_res_plot), 
             "3_cell_threshold_10pA_step_-50_140pA_500ms": (Threshold_check,
                                                            Threshold_check_plot ),
             "4_Baseline_point_by_point_response_42_points_24_x_24_grid_size_50ms_pulse_200ms_spaced_pulse_train_3_sweeps":
             (Base_line_grids,Base_line_grids_plot),
             "5_Baseline_Excitation_point_by_point_response_25"
             "_x_25_grid_size_50ms_pulse_200ms_spaced":(analysis5,plot5),
             "6_Baseline_Inhibition_point_by_point_response"
             "_25_x_25_grid_size_50ms_pulse_200ms_spaced":(analysis6,plot6),
             "7_Baseline_5_T_1_1_3_3_patterns - Copy"
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
                  "2_Input_resistance_50pA_600ms", 
                  "3_cell_threshold_10pA_step_-50_140pA_500ms",
                  "4_Baseline_point_by_point_response_42_points_24_x_24_grid_size_50ms_pulse_200ms_spaced_pulse_train_3_sweeps",
                  "5_Baseline_Excitation_point_by_point_response_25"
                  "_x_25_grid_size_50ms_pulse_200ms_spaced",
                  "6_Baseline_Inhibition_point_by_point_response"
                  "_25_x_25_grid_size_50ms_pulse_200ms_spaced",
                  "7_Baseline_5_T_1_1_3_3_patterns - Copy"
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
        print(prot_name)
        self.analysis_func = proto_map[prot_name][0]
        self.plot_func = proto_map[prot_name][1]
        self.recording = raw_trace(abf)
        self.protocol_raw_data = reader.read_raw_protocol()
        self.protol_trace = protol_trace(self.protocol_raw_data)
        self.sampling_rate = reader._sampling_rate
        self.protol_unit = self.protocol_raw_data[2][0]
        print(self.abf)
        print(self.protocol_name)

        #self.supported_obj = reader.supported_objects

class Results:
    def __init__(self,analysis,result):
        self.analysis = analysis
        self.raw_result = result


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
               pass 
    print(f"protocols assigned total protocols = {len(protocols)}")
    for p in protocols:
        print(protocols[p].protocol_name)

    # Do analysis
    for p in proto_sequence[0:4]:
        print(protocols[p].protocol_name)
        print(f"recording length = {len(protocols[p].recording)}")
        protocols[p].analysis_func(protocols[p].recording,protocols[p].sampling_rate)

    # Do plotting
    for p in proto_sequence[0:4]:
        protocols[p].plot_func(protocols[p].recording,protocols[p].sampling_rate)

if __name__  == '__main__':
    main()
