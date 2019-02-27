# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 21:59:05 2019

@author: Tim
"""

from scipy.stats.distributions import  t
import cv2
import copy
import numpy as np
from skimage.filters import threshold_minimum
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage import transform as tf
from scipy.optimize import curve_fit





def region_analyzer(I,pts,maxitter=100):
    Igg=I.copy() #going to make BW img out of green channel
    Igg[:,:,0]=Igg[:,:,1]  #red and blue get same values as green channel
    Igg[:,:,2]=Igg[:,:,1]
    thresh=threshold_minimum(Igg);  #cutoff between B & W ####MAY NEED TO CHANGE IF IT DOESNT WORK
    detect = 0
    itter =0
    while pts>detect:
        
        IBinary = (Igg[:,:,2] > thresh) #array where true at brightness > threshold
#        plt.imshow(IBinary)
        
        IBinaryFix = clear_border(1- closing(IBinary,square(3)))
#        plt.imshow(IBinaryFix)
        label_image = label(IBinaryFix)# find and lable all white contiguous shapes
        region_properties = regionprops(label_image)#get properties of labeled regions
    #    #For all the properties you can assess see: 
    #    #         http://scikit-image.org/docs/dev/api/skimage.measure.html
        n = len(region_properties) #number of objects detected
        e=np.zeros(n) #container for our eccentricities
        middle = np.zeros([n,2])
        i = 0
        dim = np.shape(I)
        tol =0.001*(dim[0]*dim[1]) #####May need to change if it doesnt work based on picture and dot size 
        param_save = np.zeros([n,4])
        for region in region_properties:
            # take regions with large enough areas
            if region.filled_area >= tol:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox  #object's bounding box
                param_save[i,:] = np.array([minr, minc, maxr, maxc])
                e[i]= region.eccentricity  #closer to 0 is closer to circle
#                    bx = (minc, maxc, maxc, minc, minc)
#                    by = (minr, minr, maxr, maxr, minr)
                y0, x0 = region.centroid
    
#                plt.plot(bx, by, '-b', linewidth=2.5)
#                plt.plot(x0,y0,'or')
                middle[i,:] = x0,y0
                i+=1  #advance 
        detect = len(np.nonzero(e)[0])
        if np.mean(IBinary)>0.5:
            thresh = thresh+0.01*thresh
        elif np.mean(IBinary<=0.5):
            thresh = thresh-0.01*thresh
        itter+=1
        if itter==maxitter:
            break
        return e,middle
    
def rank_round(e,middle,pts):
        e = e[np.nonzero(e)]
        middle = middle[np.unique(np.nonzero(middle)[0]),:]
        e_save = np.ones(pts)
        middle_save = np.zeros([pts,2])
        erank = np.sort(e)
        for i in range(0,pts,1):
            if len(e)==pts:
                e_save = e.copy()
                middle_save = middle.copy()
                break
            else:
                loc = np.where(erank[i]==e)[0]
                middle_save[i,:] = middle[loc,:]
        return e_save,middle_save

def reorder(save_loc,pts):
    store_logic = np.sum(save_loc,axis=1)
    tranform_loc = np.copy(save_loc)
#        ###THis portion of the code orders the points for transform image 
    for i in range(0,pts,1):
        order_loc = np.where(np.min(store_logic)==store_logic)
        store_logic[order_loc]= np.max(store_logic)+1000
        if i==2:
            tranform_loc[3,:]= save_loc[order_loc,:]
        elif i==3:    
            tranform_loc[2,:]= save_loc[order_loc,:]
        else:
            tranform_loc[i,:]= save_loc[order_loc,:]
    return tranform_loc
def transform_image(I,dst):
    y,x,z = np.shape(I) #image in to xyz
    src = np.array([[0, 0], [0, y-1], [x-1, y-1], [x-1, 0]]) #gives an intial guess
    tform3 = tf.ProjectiveTransform() #transfroms
    tform3.estimate(src, dst) #does the actual transformation
    warped = tf.warp(I, tform3, output_shape=(y, x)) #returns warped image 
    return warped

def filter_rgb_weight(Im,a=20,b=1,c=1):
    I_av = (a*Im[:,:,0]+b*Im[:,:,1]+c*Im[:,:,2])/(a+b+c)
    return I_av
def Image_norm(I_av,pixel_num=5000):
    Ilinear = -np.sort(np.unique(-I_av.ravel()))
    IBinary = (I_av > Ilinear[pixel_num])
    x,y = np.where(IBinary>0)
    x = x-np.min(x)
    x = x/np.max(x)
    y = y-np.min(y)
    y = y/np.max(y)
    m = (np.max(y)-np.min(y))/(np.max(x)-np.min(x))
    y = abs(y-(-m*x+1))
    return x,y

def pde_fun(x,C1,C2,C3):
    return C1*np.exp(C2*(x+C3)**2)

"""
Functions used to find diffusifity 
"""
def diffuse_fit(x,y,ti,t0):
    dt = ti-t0
    popt,pcov = curve_fit(pde_fun,x,y,p0 = [0.2  ,-1,-1])
    C1,C2,C3 = popt
    n = len(x)    # number of data points
    p = len(pcov) # number of parameters
    dof = max(0, n - p) # number of degrees of freedom
    alpha = 0.05
    # student-t value for the dof and confidence level
    tval = t.ppf(1.0 - 0.5 * alpha, dof)
    sigma = np.diag(pcov) ** 0.5
    moe = tval * sigma
    #uncertianity analysis
    D_2 = -1/(dt*4*C2)
    D_2p =1/(dt*4*(moe[1]+D_2)) 
    D_2n =1/(dt*4*(moe[1]-D_2))
    D_2u = 0.5*(abs(D_2-D_2p)+abs(D_2-D_2n))
    return D_2,D_2u,C1,C2,C3
