# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:50:04 2019

@author: Tim
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.close('all')


Function_names = ["Region_Analyzer","Rank_Round","Reorder","Transform_Image","Filter_RGB_Weight","Image_norm","Diffuse_Fit"]
timing_path_pi = "D:\Spring_2019\Final_Project\Timing_Data\Rasberrypi7_1gb.npy"
time_array_pi = np.load(timing_path_pi)*1000

timing_path_lenovo = "D:\Spring_2019\Final_Project\Timing_Data\Lenovo_i7_8gb.npy"
time_array_lenovo = np.load(timing_path_lenovo)*1000

######################################################################
#Function Time 
######################################################################
pi_average = np.mean(time_array_pi,axis=0)
pi_std = np.std(time_array_pi,axis=0)

lenovo_average = np.mean(time_array_lenovo,axis=0)
lenovo_std = np.std(time_array_lenovo,axis=0)

index = np.arange(len(Function_names))

fig,ax = plt.subplots()
bar_width = 0.35
opacity = 0.4
error_config = {'ecolor': '0.3'}
rects1 = ax.bar(index, pi_average, bar_width,
                alpha=opacity, color='b',
                yerr=pi_std, error_kw=error_config,
                label='Pi')

rects2 = ax.bar(index + bar_width, lenovo_average, bar_width,
                alpha=opacity, color='r',
                yerr=lenovo_std, error_kw=error_config,
                label='Lenovo')
ax.set_ylabel('Time (ms)')
ax.set_xlabel('Functions')
ax.set_title("Python Computational Time Cost")
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels((Function_names))
ax.legend()
params = {'axes.labelsize': 15,'axes.titlesize':15,  'legend.fontsize': 10, 'xtick.labelsize': 6, 'ytick.labelsize': 10}
matplotlib.rcParams.update(params)
fig.tight_layout()
plt.savefig('./comptime_bargraph.jpg',dpi=300)
######################################################################
#Total Time 
######################################################################

pi_sumaverage = np.mean(np.sum(time_array_pi,axis=1))
pi_sumstd = np.std(np.sum(time_array_pi,axis=1))

lenovo_sumaverage = np.mean(np.sum(time_array_lenovo,axis=1))
lenovo_sumstd = np.std(np.sum(time_array_lenovo,axis=1))

index = np.arange(1)

fig,ax = plt.subplots()
bar_width = 0.35
opacity = 0.4
error_config = {'ecolor': '0.3'}
rects1 = ax.bar(index, pi_sumaverage, bar_width,
                alpha=opacity, color='b',
                yerr=pi_sumstd, error_kw=error_config,
                label='Pi')

rects2 = ax.bar(index + bar_width+0.1, lenovo_sumaverage, bar_width,
                alpha=opacity, color='r',
                yerr=lenovo_sumstd, error_kw=error_config,
                label='Lenovo')
ax.set_ylabel('Time (ms)')
ax.set_xlabel('Device')
ax.set_title("Python Total Computational Time Cost")
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax.legend()
params = {'axes.labelsize': 15,'axes.titlesize':15,  'legend.fontsize': 10, 'xtick.labelsize': 6, 'ytick.labelsize': 10}
matplotlib.rcParams.update(params)
fig.tight_layout()
plt.savefig('./comptotaltime_bargraph.jpg',dpi=300)

