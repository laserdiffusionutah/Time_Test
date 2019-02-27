# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:46:37 2019

@author: Tim
"""
import time 
import os
import numpy as np
from laser_diffusion_alpha import region_analyzer,rank_round,reorder,\
     transform_image,filter_rgb_weight,Image_norm,diffuse_fit


"""
Function finds the dots by specifiying the number of dots in the given image
"""
path = './Trial_1'
basename = 'Trial_1_'
len_base = len(basename)
pts = 4

dire_list = os.listdir(path)
timed_functions = 7
timing_path = './Timing_Data'
timing_name = 'Lenovo_i7_8gb.npy'
timingjoin = os.path.join(timing_path,timing_name)
timing_save = np.zeros([len(dire_list),timed_functions])


for i in range(0,len(dire_list),1):
    loadpath =os.path.join(path,dire_list[i])
    time_extractor = float(dire_list[0][len(basename):-4].replace('_','.'))
    I = np.load(loadpath)
    
    t0 = time.time()
    e,middle = region_analyzer(I,pts,maxitter=100)
    tf = time.time()
    timing_save[i,0] = tf-t0
        
    t0 = time.time()
    e_round, middle_round = rank_round(e,middle,pts) 
    tf = time.time()
    timing_save[i,1] = tf-t0
    del e,middle
    
    t0 = time.time()
    transform = reorder(middle_round,pts)
    tf = time.time()
    timing_save[i,2] = tf-t0
    del  e_round, middle_round 

#    transform_image(I,transform)
#        ###THis portion of the code orders the points for transform image
    t0 = time.time()
    Im  = transform_image(I,transform)
    tf = time.time()
    timing_save[i,3] = tf-t0
    del I,transform

    t0 = time.time()
    I_av = filter_rgb_weight(Im)
    tf = time.time()
    timing_save[i,4] = tf-t0
    del Im
    
    t0 = time.time()
    x,y = Image_norm(I_av)
    tf = time.time()
    timing_save[i,5] = tf-t0
    del I_av

    t0 = time.time()
    D_2,D_2u,C1,C2,C3 = diffuse_fit(x,y,time_extractor,0)
    tf = time.time()
    timing_save[i,6] = tf-t0
    
    time.sleep(1)
    print(i)
    
    
np.save(timingjoin,timing_save)

    
    
    
    
