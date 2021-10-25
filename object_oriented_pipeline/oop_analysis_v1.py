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
from matplotlib import gridspec


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

def list_patterns(p):
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
        brightidx2D = np.transpose(np.unravel_index(brightidx-1, (num_of_Rows,num_of_Cols))) #Getting     2d coordinates from linear coord
        for j in range(len(brightidx2D)):
            zeromatrix[brightidx2D[j][0]][brightidx2D[j][1]] = 1 #Replacing the zeors with 1s
        Framedict[fr_no] = zeromatrix

#    print(Framedict)
    return Framedict

def plot_projections(p, projections):
    p = str(p)
    for ind,i in enumerate(projections):
        for ind_l,l in enumerate(i):
            fr = i[l]
            plt.imshow(fr)
#            plt.show()
            plt.pause(0.05)
            plt.savefig(f"{p}_{ind}_{ind_l}.png")
            plt.close()


def plot_base(fig, panelTitle, plotPos, xlabel, ylabel,ylim, fig_text ):
    spec = gridspec.GridSpec(ncols=2, nrows=2,
                         width_ratios=[2, 1], wspace=0.5,
                         hspace=0.5, height_ratios=[1, 2])
    x,y = 3,7
    ax = fig.add_subplot(x, y , plotPos)
#    ax.set_aspect()
#    ax.set_adjustable("datalim")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.text(0.2,1.2, panelTitle, fontsize=15, transform=ax.transAxes)
    ax.text(0,-0.7, fig_text, fontsize=12, transform=ax.transAxes)
    ax.set_ylim(ylim)
    return ax




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

def analysis5( protocol , andor ):
    # do something
    print( "analysis3" )


def Ttl_dict(fig, recording,sampling_rate, pos):
    y_max_mean = []
    y_min_mean = []
    thresh_state =0
    sampling_rate = sampling_rate
    rows = 3
    va_,vb_,vc_ = [],[],[]
    ta_,tb_,tc_ = [],[],[]
    va_mean,vb_mean,vc_mean = [],[],[]
    ta_mean,tb_mean,tc_mean = [],[],[]
    na,nb,nc = 0,0,0
    for ind_i,i in enumerate(recording):
        r = ind_i%rows
        a,b,c =(1%3,2%3,0)
        v = recording[ind_i][1]
        t = recording[ind_i][0]
        if r == a:
            panelTitle_a = "cell recording"
            plotPos_a = (pos)
            xlabel_a = "time(s)"
            ylabel_a = "mV"
            fig_text_a = "."
            na +=1
            ta_.append(t)
            va_.append(v)
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
            if na == 3:
                va_mean = np.mean(va_, axis = 0)
                ta_mean = np.mean(ta_, axis = 0)
                y_max = max(va_mean)+5
                y_min = min(va_mean)-5
                ylim_a = (y_min,y_max)
        elif r == b:
            panelTitle_b = "photodiode"
            plotPos_b = (pos+7)
            xlabel_b = "time(s)"
            ylabel_b = "pA"
            fig_text_b = " "
            nb +=1
            vb_.append(v)
            tb_.append(t)
            if nb == 3:
                vb_mean = np.mean(vb_, axis = 0)
                tb_mean = np.mean(tb_, axis = 0)
                y_max = max(vb_mean)+5
                y_min = min(vb_mean)-5
                ylim_b = (y_min,y_max)
                #                print(tb_,vb_)
        elif r == c:
            panelTitle_c = "ttl"
            plotPos_c = (pos+14)
            xlabel_c = "time(s)"
            ylabel_c = "v"
            fig_text_c = " "
            nc +=1
            vc_.append(v)
            tc_.append(t)
            if nc == 3:
                vc_mean = np.mean(vc_, axis = 0)
                tc_mean = np.mean(tc_, axis = 0)
                y_max = max(vc_mean)+5
                y_min = min(vc_mean)-5
                ylim_c = (y_min,y_max)
    axa = plot_base(fig, panelTitle_a, plotPos_a,
              xlabel_a, ylabel_a, ylim_a, fig_text_a)
                                                                                                   
    for i,ind in enumerate(va_):                                                                  
        axa.plot(ta_[i],va_[i], alpha = 0.3)                                                     
        axa.plot(ta_mean, va_mean, color = 'r')
        axa.scatter(peaks_t, peaks_v, alpha=0.5, color='r')                                            
        axb = plot_base(fig, panelTitle_b, plotPos_b,                                                
                        xlabel_b, ylabel_b, ylim_b, fig_text_b)                                     
    for i,ind in enumerate(vb_):
        axb.plot(tb_[i],vb_[i],alpha=0.3)                                                      
        axb.plot(tb_mean, vb_mean, color = 'k')                                               
        axc = plot_base(fig, panelTitle_c, plotPos_c,                                        
                        xlabel_c, ylabel_c, ylim_c, fig_text_c)
    for i,ind in enumerate(vc_):                                                                   
        axc.plot(tc_[i],vc_[i],alpha = 0.3)                                                            
        axc.plot(tc_mean, vc_mean, color = 'b')                                                             

def analysis6( protocol , andor ):
    # do something
    print( "analysis3" )

def analysis7( protocol , andor ):
    # do something
    print( "analysis3" )

def analysis8( protocol , andor ):
    # do something
    print( "analysis3" )

def analysis9( protocol , andor ):
    # do something
    print( "analysis3" )


def analysis10( protocol , andor ):
    # do something
    print( "analysis3" )


def analysis11( protocol , andor):
    # do something
    print( "analysis2" )

def analysis12( protocol , andor):
    # do something
    print( "analysis3" )

def analysis13( protocol , andor):
    # do something
    print( "analysis2" )

def analysis14( protocol , andor):
    # do something
    print( "analysis3" )       


def Rmp_plot(fig, recording,sampling_rate, pos):
     r = recording[0]
     t = r[0]
     v = r[1]
     y_max = max(v)+5
     y_min = min(v)-5
     ylim = (y_min,y_max)
     title = "Resting membrane potential"
     xlabel = "time(s)"
     ylabel = "membrane volatge (mV)"
     plotPos = (pos)
     rmp = np.around(np.average(v), decimals = 2)
     fig_text = "resting membrane potential= "+str(rmp)+" mV"
     ax = plot_base(fig,title,plotPos,xlabel,ylabel, ylim, fig_text)
     ax.plot(t,v)
     print( "analysis1" )

def Series_res_plot(fig, recording, sampling_rate, pos ):
    print(f"length of recording in series res check {len(recording)}")
    mean_R = []
    y_max_mean = []
    y_min_mean = []
    title = "Series resistance"
    xlabel = "time(s)"
    ylabel = "membrane volatage (mV)"
    plotPos = (pos)
    sampling_rate = sampling_rate
    for i, r in enumerate(recording):
        r = recording[i]
        t = r[0]
        v = r[1]
        y_max = max(v)
        y_min = min(v)
        trace = v
        time = t
        y_max_mean.append(y_max)
        y_min_mean.append(y_min)
        Vl= np.mean(trace[int(sampling_rate*0.5):int(sampling_rate*0.55)])
        Vb= np.mean(trace[int(sampling_rate*0.15):int(sampling_rate*0.20)])
        tl = time[int(sampling_rate*0.5)]
        tb = time[int(sampling_rate*0.15)]
        print (f"reference points = {Vl, Vb}")
        input_R = (((Vl-Vb)*1000)/(-50))
        print(f"input_resistance = {input_R} MOhm")
        mean_R.append(input_R)
    y_max_mean = np.mean(y_max_mean)
    y_min_mean = np.mean(y_min_mean)
    ylim = (y_min_mean, y_max_mean)
    mean_R =np.round(np.mean(mean_R), decimals=2)
    input_R = (mean_R - 3.5)
    fig_text = "series resitance = "+ str(mean_R) + " MOhm"
    fig_text = "input resistance resitance = "+ str(input_R) + " MOhm"
    ax = plot_base(fig, title,plotPos,xlabel,ylabel,ylim,fig_text)
    ax.plot(t,v)
    ax.scatter(tl,Vl,marker='o',color='r')
    ax.scatter(tb,Vb,marker='o',color='k')
    print (f" series_res = {mean_R} input res = {input_R}" )

def Threshold_check_plot( fig, recording,sampling_rate, pos):
    print(f"length of recording in threshold check {len(recording)}")
    y_max_mean = []
    y_min_mean = []
    thresh_state =0
    sampling_rate = sampling_rate
    title = "Cell threshold "
    xlabel = "time(s)"
    ylabel = "membr2ane volatage (mV)"
    plotPos = (pos)
    for i, r in enumerate(recording):
        r = recording[i]
        t = r[0]
        v = r[1]
        y_max = max(v)
        y_min = min(v)
        trace = v
        time = t
        y_max_mean.append(y_max)
        y_min_mean.append(y_min)
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
                x,y = t,v
                peak_t,peak_v = time[peaks[0]-10],trace[peaks[0]-10]
                thresh_t,thresh_v = time[t_dt_max],trace[v_dt_max]
                plt.scatter(time[peaks[0]-10],trace[peaks[0]-10], color='r')
                plt.scatter(time[t_dt_max],trace[v_dt_max], color='k')
                print(f"max dv_dt locci = {v_dt_max , t_dt_max}")
    y_max_mean = max(y_max_mean)+10
    y_min_mean = min(y_min_mean)-10
    ylim = (y_min_mean, y_max_mean)
    plt.ylim = (y_min-10, y_max+10)
    fig_text = "thrshold voltage =  mV"
    ax = plot_base(fig,title,plotPos,xlabel,ylabel,ylim,fig_text)
    ax.plot(x,y)
    ax.scatter(peak_t,peak_v, color='r')
    ax.scatter(thresh_t,thresh_v, color='k')

def optical_plot(fig, recording,sampling_rate, pos):
    print(recording)
    y_max_mean = []
    y_min_mean = []
    thresh_state =0
    sampling_rate = sampling_rate
    rows = 3
    va_,vb_,vc_ = [],[],[]
    ta_,tb_,tc_ = [],[],[]
    va_mean,vb_mean,vc_mean = [],[],[]
    ta_mean,tb_mean,tc_mean = [],[],[]
    na,nb,nc = 0,0,0
    for ind_i,i in enumerate(recording):
        r = ind_i%rows
        a,b,c =(1%3,2%3,0)
        v = recording[ind_i][1]
        t = recording[ind_i][0]
        if r == a:
            panelTitle_a = "cell recording"
            plotPos_a = (pos)
            xlabel_a = "time(s)"
            ylabel_a = "mV"
            fig_text_a = "."
            na +=1
            ta_.append(t)
            va_.append(v)
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
            if na == 3:
                va_mean = np.mean(va_, axis = 0)
                ta_mean = np.mean(ta_, axis = 0)
                y_max = max(va_mean)+5
                y_min = min(va_mean)-5
                ylim_a = (y_min,y_max)
        elif r == b:
            panelTitle_b = "photodiode"
            plotPos_b = (pos+7)
            xlabel_b = "time(s)"
            ylabel_b = "pA"
            fig_text_b = " "
            nb +=1
            vb_.append(v)
            tb_.append(t)
            if nb == 3:
                vb_mean = np.mean(vb_, axis = 0)
                tb_mean = np.mean(tb_, axis = 0)
                y_max = max(vb_mean)+5
                y_min = min(vb_mean)-5
                ylim_b = (y_min,y_max)
                #                print(tb_,vb_)
        elif r == c:
            panelTitle_c = "ttl"
            plotPos_c = (pos+14)
            xlabel_c = "time(s)"
            ylabel_c = "v"
            fig_text_c = " "
            nc +=1
            vc_.append(v)
            tc_.append(t)
            if nc == 3:
                vc_mean = np.mean(vc_, axis = 0)
                tc_mean = np.mean(tc_, axis = 0)
                y_max = max(vc_mean)+5
                y_min = min(vc_mean)-5
                ylim_c = (y_min,y_max)
    axa = plot_base(fig, panelTitle_a, plotPos_a,
              xlabel_a, ylabel_a, ylim_a, fig_text_a)

    for i,ind in enumerate(va_):
        axa.plot(ta_[i],va_[i], alpha = 0.3)
    axa.plot(ta_mean, va_mean, color = 'r')
    axa.scatter(peaks_t, peaks_v, alpha=0.5, color='r')
    axb = plot_base(fig, panelTitle_b, plotPos_b,
                    xlabel_b, ylabel_b, ylim_b, fig_text_b)
    for i,ind in enumerate(vb_):
        axb.plot(tb_[i],vb_[i],alpha=0.3)
    axb.plot(tb_mean, vb_mean, color = 'k')
    axc = plot_base(fig, panelTitle_c, plotPos_c,
              xlabel_c, ylabel_c, ylim_c, fig_text_c)
    for i,ind in enumerate(vc_):
        axc.plot(tc_[i],vc_[i],alpha = 0.3)
    axc.plot(tc_mean, vc_mean, color = 'b')

#    ax1.scatter(peaks_t,peaks_v, marker='.')


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


proto_map = { "1_RMP_gap_free_1min": ( Rmp, Rmp_plot, 1 ), 
             "2_Input_resistance_-20pA_350ms": (Series_res, Series_res_plot, 8), 
             "3_cell_threshold_10pA_step_-50_140pA_500ms": (Threshold_check,
                                                            Threshold_check_plot, 15),
             "4_Baseline_point_by_point_response_42_points_24_x_24_grid_size_50ms_pulse_200ms_spaced_pulse_train_3_sweeps":
             (Base_line_grids,optical_plot, 2),
             "5_Excitation_point_by_point_response_42_points_24_x_24_grid_size_50ms_pulse_200ms_spaced_pulse_train_3_sweeps":(analysis5, optical_plot, 3),
             "6_Inhibition_point_by_point_response_42_points_24_x_24_grid_size_50ms_pulse_200ms_spaced_pulse_train_3_sweeps - Copy":(analysis6,optical_plot,4),
             "7_Baseline_5_T_1_1_3_3_patterns - Copy":(analysis7, optical_plot, 5) ,
#             "8_10_Pattern_baseline_excitation_0mV_holding"
#             "_25X25_single_point":(analysis8,optical_plot),
             "9_Inhibition_5_T_1_1_3_3_patterns - Copy":(analysis9,optical_plot, 6),
             "10_Training_5_T_1_1_3_3_patterns":(analysis10,optical_plot, 7),
#             "11_Inhibition_point_by_point_response_25_x_25"
#             "_grid_size_50ms_pulse_200ms_spaced":(analysis11,optical_plot),
#             "12_Inhibition_point_by_point_response_25_x_25"
#             "_grid_size_50ms_pulse_200ms_spaced":(analysis12,optical_plot),
#             "13_10_Pattern_Inhibition-70mV_holding_25X25_single_point":(analysis13,optical_plot),
#             "14_10_Pattern_Inhibition-70mV_holding_25X25_single_point":(analysis14,optical_plot)

            }

proto_sequence = ["1_RMP_gap_free_1min", 
                  "2_Input_resistance_-20pA_350ms", 
                  "3_cell_threshold_10pA_step_-50_140pA_500ms",
                  "4_Baseline_point_by_point_response_42_points_24_x_24_grid_size_50ms_pulse_200ms_spaced_pulse_train_3_sweeps",
                  "5_Excitation_point_by_point_response_42_points_24_x_24_grid_size_50ms_pulse_200ms_spaced_pulse_train_3_sweeps",
                  "6_Inhibition_point_by_point_response_42_points_24_x_24_grid_size_50ms_pulse_200ms_spaced_pulse_train_3_sweeps - Copy",
                  "7_Baseline_5_T_1_1_3_3_patterns - Copy",
#                  "8_10_Pattern_baseline_excitation_0mV_holding"
#                  "_25X25_single_point",
                  "9_Inhibition_5_T_1_1_3_3_patterns - Copy",
                  "10_Training_5_T_1_1_3_3_patterns",
#                  "11_Inhibition_point_by_point_response_25_x_25"
#                  "_grid_size_50ms_pulse_200ms_spaced",
#                  "12_Inhibition_point_by_point_response_25_x_25"
#                  "_grid_size_50ms_pulse_200ms_spaced",
#                  "13_10_Pattern_Inhibition-70mV_holding_25X25_single_point",
#                  "14_10_Pattern_Inhibition-70mV_holding_25X25_single_point"
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
        self.plot_pos = proto_map[prot_name][2]
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
    parser.add_argument('--folder_files', '-f'
            , required = False, default ='./', type=str
            , help = 'path of folder with  abf files '
            )
    parser.add_argument('--folder_patterns', '-p'
            , required = False, default ='./', type=str
            , help = 'path of folder with grid info text'
            )
    args = parser.parse_args()
    fpaths = list_files( args.folder_files )
#    print(fpaths)
    ppaths = list_patterns( args.folder_patterns )
#    print(ppaths)
#    print(fpaths)
#    projections = []
#    for p in ppaths:
#        a = load_text(p)
#        projections.append(a)
#    plot_projections(p, projections)

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
    
    for p in proto_sequence:
        print(protocols[p].protocol_name)
        print(protocols[p].plot_pos)
        protocols[p].analysis_func(protocols[p].recording,protocols[p].sampling_rate)

    # Do plotting
    fig = plt.figure(figsize = (30,10))
    for p in proto_sequence:
        print(f"ploting file {protocols[p].abf}")
        pos = protocols[p].plot_pos
        try:
            protocols[p].plot_func(fig, protocols[p].recording,
                                   protocols[p].sampling_rate, pos)
        except:
            pass
    fig.tight_layout()
    plt.show()

if __name__  == '__main__':
    main()
