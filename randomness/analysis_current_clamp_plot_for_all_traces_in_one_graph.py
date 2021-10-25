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
import neo.io as nio
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor as extractor

class Args: pass 
args_ = Args()


f_list = []
print(f_list,'...')
def main(**kwargs):
    p = Path(kwargs['folder_path'])
    f_list=list(p.glob('**/*.dat'))
    outdir = p/'results'
    outdir.mkdir(exist_ok=True, parents=True)
    for fi, f in enumerate(f_list):
        print(f"[info] Analysing file {f.name}")
        itr_no = 0
        seg_no = 0
        Vm_trail = []
        thresh_state = 0
        f_path = f
        f = str(f)
        segments = nio.StimfitIO(f).read_block().segments
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)           
        for si, segment in  enumerate(segments):
            seg_no +=1
            fI = 10000*fi + si
            #print('analysing the segment ',segment)
            analog_signals = segment.analogsignals
            #print(analog_signals)
            for ti, trace in enumerate(analog_signals):
                itr_no += 1
                v = trace
                v = np.ravel(v)
                if '1.0 pA' == str(v.units):
                    print('x 1pA')
                    continue
                if np.isnan(v)[0] == True:
                    print('x Vnan')
                    continue
                if itr_no != 1:
                    traces = 'traces compiled'
                else:
                    traces = 'trace_alone'
                print(f'trace {ti}', end=', ')
                protocol_type = 'Current_clamp'
                v = v.magnitude
                Vm_trace = np.mean(v[len(v)-300:len(v)])
                tf = trace.t_stop
                ti = trace.t_start
                t = np.linspace(0,float(tf - ti), len(v))
                ax.plot(t,v,label = 'trace numer = '+str(itr_no))

                try:
                 Trace_with_features = extractor(t=t, v=v, filter = float(trace.sampling_rate)/2500,min_peak=-20.0, dv_cutoff=20.0, max_interval=0.005, min_height=2.0, thresh_frac=0.05, baseline_interval=0.1, baseline_detect_thresh=0.3, id=None)
                 Trace_with_features.process_spikes()
                 neuron_threshold_v = Trace_with_features.spike_feature("threshold_v")
                 if thresh_state == 0 and len(neuron_threshold_v) >=1:
                     neuron_threshold_v = Trace_with_features.spike_feature("threshold_v")[0]
                     neuron_threshold_t = Trace_with_features.spike_feature("threshold_t")[0]
                     trough_v = Trace_with_features.spike_feature("trough_v")[0]
                     trough_t = Trace_with_features.spike_feature("trough_t")[0]
                     ax.plot(neuron_threshold_t,neuron_threshold_v,'o', color ='k',label = 'threshold voltage')
                     ax.plot(trough_t,trough_v,'o', color ='r',label = 'trough_v')
                     ax.figtext(1, 0.20, "trough_v = "+ str(np.around(trough_v, decimals = 2))+"mV")
                     ax.figtext(1, 0.15, "trough_t = "+ str(np.around(trough_t, decimals = 2))+'s')
                     threshold_state = 1
                except Exception as e:
                     print('Could not be plotted:', e)
        ax.legend()
        ax.set_ylabel('signal amplitude in '+str(trace[0]).split()[1])
        ax.set_xlabel('time (s)')
        ax.set_title('recording type: '+str(trace.name).split("-")[0]+' '+str(len(segments))+' '+traces+ ' of segment number '+ str(seg_no))
        outfile = str(outdir)+"/"+str(f_path.stem)+" "+f'{fI}_compiled.png'
        plt.savefig(outfile)
        print('---> Saved to %s' % outfile)
        fig = plt.close()



#print(f"[info] folder path{f_lif_list}")
#print(f_list)
        
 




















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
