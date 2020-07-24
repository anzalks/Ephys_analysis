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
from scipy import signal as ssig
import matplotlib.animation as animation

class Args: pass 
args_ = Args()

def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

f_list = []
print(f_list,'...')
def main(**kwargs):
    p = Path(kwargs['folder_path'])
    f_list=list(p.glob('**/*dat'))
    outdir = p/'results_animated'
    outdir.mkdir(exist_ok=True, parents=True)
    for fi, f in enumerate(f_list):
        print(f"[info] Analysing file {f.name}")
        itr_no = 0
        seg_no = 0
        Vm_trail = []    
        f_path = f
        f = str(f)
        segments = nio.StimfitIO(f).read_block().segments
        for si, segment in  enumerate(segments):
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
                v = ssig.resample(v,5000) 
                tf = trace.t_stop
                ti = trace.t_start
                t = np.linspace(0,float(tf - ti), len(v))
                t = ssig.resample(t,5000)
                data = np.array([t,v])
                fig1 = plt.figure()
                l, = plt.plot([], [], 'r-')
                plt.xlim(0, 1.2)
                plt.ylim(-100,60)
                plt.xlabel('time(s)')
                plt.ylabel('signal amplitude (mV)')
                plt.title('training the slice')
                line_ani = animation.FuncAnimation(fig1, update_line, 5000, fargs=(data, l),
                                   interval=1, blit=True)
                line_ani.save(f'{outdir}/{si}_{ti}_lines.mp4', writer=writer)
                plt.close()

 

#                ax.plot(t,v,label = 'trace numer = '+str(itr_no),animated=True)
#        ax.legend()
#        ax.set_ylabel('signal amplitude in '+str(trace[0]).split()[1])
#        ax.set_xlabel('time (s)')
#        #ax.set_ylim([-1,2])
#        #ax.set_xlim([2,10])
#        ax.set_title('recording type: '+str(trace.name).split("-")[0]+' '+ str(seg_no)+' of  '+'traces')
#        outfile = str(outdir)+"/"+str(f_path.stem)+" "+f'{fI}_compiled.svg'
#        plt.savefig(outfile, bbox_inches = "tight")
#        print('---> Saved to %s' % outfile)
#        fig = plt.close()
#        
#        itr_no = 0
#        seg_no = 0
#        thresh_state = 0
#        for si, segment in  enumerate(segments):
#            seg_no +=1
#            fI = 10000*fi + si
#            #print('analysing the segment ',segment)
#            analog_signals = segment.analogsignals
#            #print(analog_signals)
#            for ti, trace in enumerate(analog_signals):
#                fig1 = plt.figure(figsize=(8,5))
#                ax1 = fig1.add_subplot(111)
#                TI = 100001*fI+ti
#                itr_no += 1
#                v = trace
#                v = np.ravel(v)
#                if '1.0 pA' == str(v.units):
#                    print('x 1pA')
#                    continue
#                if np.isnan(v)[0] == True:
#                    print('x Vnan')
#                    continue
#                if itr_no != 1:
#                    traces = 'traces compiled'
#                else:
#                    traces = 'trace_alone'
#                print(f'trace {ti}', end=', ')
#                protocol_type = 'Current_clamp'
#                trace_unit = str(v.units).split('.')[1]
#                v = v.magnitude
#                tf = trace.t_stop
#                ti = trace.t_start
#                t = np.linspace(0,float(tf - ti), len(v))
#                Vm_i = np.mean(v[0:299])
#                Vm_f = np.mean(v[len(v)-300:len(v)])
#                del_Vm = Vm_f-Vm_i
#                ax1.plot(t,v,label = 'trace numer = '+str(itr_no))
#                ax1.legend()
#                ax1.set_ylabel('signal amplitude in '+str(trace[0]).split()[1])
#                ax1.set_xlabel('time (s)')
#                #ax1.set_ylim([-1,2])
#                #ax1.set_xlim([2,10])
#                ax1.set_title('recording type: '+str(trace.name).split("-")[0]+': single trace')
#                outfile = str(outdir)+"/"+str(f_path.stem)+" "+f'{TI}.svg'
#                plt.savefig(outfile, bbox_inches = "tight")
#                print('---> Saved to %s' % outfile)
#                fig1 = plt.close()
#



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
